import re
import time, operator, subprocess
from typing import Tuple, List, NamedTuple, Dict
from warnings import warn
from functools import reduce
from numpy import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, datasets, backend, losses
import matplotlib.pyplot as plt
import PIL
from tensorflow.python.keras.callbacks import History

MB=1024**2 ; GB=1024**3
class GPU_Ram: TeslaK80_1Proc= 12*GB; TeslaV100=16*GB; TeslaP100=16*GB; TeslaM60=16*GB; NoGPU=128*MB


class HistoryAndTimeItem(NamedTuple):
    index:int ; num_epochs:int ; time_secs:float ; batch_size:int ; history: Dict[str, float]

class HistoryAndTime(List[HistoryAndTimeItem]):
    def __init__(self):
        super(HistoryAndTime, self).__init__()
    def num_epochs(self)->int       : return sum([e.num_epochs for e in self])
    def total_time(self)->float     : return sum([e.time_secs for e in self])
    def __repr__(self):
        repr="HistoryAndTime(num_epochs={}, total_time={},latest metrics:{}\n{}"\
            .format(
                self.num_epochs(), self.total_time(), self[-1].history.keys(),
                "\n".join(
                    [ "{}={}".format(k, hi[k]) for hi in [i.history for i in self] for k in hi.keys() ]
                ))
        return repr


def main(gpu_ram= GPU_Ram.TeslaK80_1Proc, epochs=6 ) -> (models.Sequential, HistoryAndTime):

    mnist_4Dense_net= models.Sequential([
            layers.Reshape(target_shape= (28*28,),input_shape=(28, 28)),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='relu')]
            )
    mnist_4Dense_net.summary()
    ds_train,ds_val=create_mnist_datasets()
    best_batch_size=max_batch_size(gpu_ram,mnist_4Dense_net,default_max=64)
    history_and_time= train_and_eval(mnist_4Dense_net,
                                     ds_train, ds_val,
                                     num_epochs=epochs,
                                     validation_freq=1,
                                     batch_size=best_batch_size)
    return (mnist_4Dense_net, history_and_time)


@tf.function
def mnist_to_float32_int(x,y)-> Tuple[tf.Tensor,tf.Tensor]:
    return tf.cast(x,tf.float32)/255.0, tf.cast(y,tf.int32)


def create_mnist_datasets(shuffle_buffer=40000)->Tuple[tf.data.Dataset, tf.data.Dataset]:

    (x_train, y_train), (x_val, y_val)= datasets.mnist.load_data()
    ds_train=tf.data.Dataset.from_tensor_slices( (x_train,y_train)) \
                .map(mnist_to_float32_int).shuffle(shuffle_buffer)
    ds_val=tf.data.Dataset.from_tensor_slices((x_val,y_val)) \
                .map(mnist_to_float32_int).shuffle(shuffle_buffer)
    return ds_train,ds_val


def max_batch_size(gpu_ram_bytes:int,
                   model:models.Model, scalar_width:int=4,
                   default_max:int=32,
                   usable=0.95, verbose=True)->int:
    """
    See https://www.microsoft.com/en-us/research/uploads/prod/2020/05/dnnmem.pdf for
    more complications than are dealt with here; and for a proposal for a tool to do away
    with this estimation.
    See https://arxiv.org/1609.04836 for the suggestion that anything over 32 is probably
    bad anyway.
    :param gpu_ram_bytes: The RAM available to your graphics processor. For dual-GPU cards
    this should still be the memory of a single GPU unless you configure for multiple workers
    :param model: a keras Model which can be inspected to find it's weights and inputs
    :param scalar_width: the width of your datatype in bytes. e.g. 4 for float32, 8 for float64
    :param default_max: Cut-off beyond which we assume that bigger batches will
    degrade generalisability (e.g. https://arxiv.org/1609.04836)
    :param usable: defaults to 0.95 The fraction of GPU memory that should be considered available
    for your model and inputs. Usually less than 100% because of framework, alignment loss,
    buffers, runtime context etc.
    :param verbose: print calculation
    :return: an integer which is our best guess for the biggest power of 2 batch size that will
    fit into your GPU memory at one go.
    """
    assert 0 < gpu_ram_bytes, 'required: 0 < gpu_ram_bytes, you said %r' % gpu_ram_bytes
    assert 0 < usable, 'required: 0 < usable, you said %r' % usable
    assert 0 < scalar_width, 'required: 0< model_dtype_width, you said %r' % scalar_width
    assert model and model.layers, 'model.layers must not be None or empty'
    warnif(usable>1, "You've set usable GPU memory usage to more than 100%")
    all_inputs = sum([ reduce(operator.mul,[dim if dim else 1 for dim in l.input_shape])
                        for l in model.layers])
    outputs = reduce(operator.mul,
                     [dim if dim else 1 for dim in model.layers[-1].output_shape])
    labels = outputs
    tensors_size= all_inputs + outputs + labels
    num_ephemeral=tensors_size # Actual value is ‘we have no idea, it depends on implementation’
    num_weights=sum(
            [ a.shape.num_elements()
              for a in model.trainable_weights + model.non_trainable_weights ])
    num_gradients=num_weights
    num_scalars=tensors_size + num_weights + num_gradients + num_ephemeral

    max_size= int(usable * gpu_ram_bytes / scalar_width / num_scalars)
    best_size= min( 2**int(math.floor(math.log(max_size, 2))), default_max)
    best_size=max(1,best_size)
    if verbose:
        print('Calculated tensors_size={}, num_weights={}, scalar width={}, '
              'max batch size for {}GB is {}, best size is {}'\
              .format(tensors_size,num_weights,scalar_width,gpu_ram_bytes/GB,max_size,best_size))
    return best_size


def decay_pow2(current_step, num_steps, start, end=1, curve='exponential'):
    """
    Decay from start to end, rounding to powers of two
    :param current_step: we expect current_step between 0 and num_steps
    :param num_steps:
    :param start: The value wanted when current_step==0
    :param end: default to 1. The value wanted when current_step==num_steps
    :param curve: linear or exponential
    :return: a power of 2 representing the fraction current_step/num_steps of the way from
    start to end along the given curve
    """
    linear_fraction=current_step/num_steps
    if curve=='exponential':
        log2end=math.log2(end)
        log2start=math.log2(start)
        progress= log2end * (1 - linear_fraction) + log2start * linear_fraction
        exp= log2start -log2end - progress
        return max(1, 2**int(exp))
    else:
        linear=start * (1 - linear_fraction) + end * linear_fraction
        return 0 if linear==0 else 2**int(0.5 + math.log2(linear))


def train_and_eval(model: models.Model,
                   ds_train: tf.data.Dataset, ds_val: tf.data.Dataset,
                   num_epochs, validation_freq=1,
                   save_to_path='TrainedModel',
                   batch_size=32,
                   batch_size_decay=None):
    """
    train and eval the model and return a
    :param model: the model to train
    :param ds_train: the training dataset
    :param ds_val: the validation dataset
    :param num_epochs: total number of epochs to train
    :param validation_freq: how often to validate
    :param save_to_path: path to save the trained model
    :param batch_size: the batch size to use or, if use batch size decay,
    the initial batch size
    :param batch_size_decay: None or 'exponential' or 'linear'
    :return: a HistoryAndTime containing timings and the Tensorflow history dict
    """
    print('compiling...')
    model.compile(
            optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy()])
    try: subprocess.run('nvidia-smi')
    except Exception: pass

    print('training...')
    historyandtime= HistoryAndTime()
    start=time.time()
    batch_size= batch_size if not batch_size_decay \
                           else decay_pow2(epoch - 1, num_epochs, batch_size, batch_size_decay)
    ds_batch=ds_train.batch(batch_size)
    history=model.fit(ds_batch,
                      epochs=num_epochs,
                      validation_data=ds_val.batch(batch_size),
                      validation_freq=validation_freq)
    tf.keras.models.save_model(model,save_to_path)
    elapsed = time.time() - start
    historytime_item=HistoryAndTimeItem(1, num_epochs, elapsed, batch_size, history.history)
    historyandtime.append(historytime_item)
    epochs_descr='Epochs {}-{} batch size {} '.format(1, num_epochs, batch_size)
    print(epochs_descr,'took {:.1f}sec. Validation loss= {:.2f}, Validation Accuracy= {}%'\
            .format( historytime_item.time_secs,
                     historytime_item.history['val_loss'][-1],
                     int(100*historytime_item.history['val_sparse_categorical_accuracy'][-1])))
    return historyandtime

def warnif(condition:bool, message:str, **kwargs):
    if condition: warn(message,**kwargs)


def plot_metrics(metrics: Dict[str,float],
                 what_to_plot: Dict[str,list],
                 legends_based_on=(('Validation','^val_'),('Train','.*'),),
                 plotted_against='epoch',
                 legend_loc='upper left'):
    """
    :param metrics: Assumed to be a tensorflow.keras.callbacks.History.history -
    i.e. a dict[str,list] where the key is the name of a metric, the list contains float values,
    and the list is assumed to be index on epoch.
    :param what_to_plot: One graph will be generated for each key. The list is assumed to
    contain names matching the keys of metrics.
    :param legends_based_on: Use this to categorise each metric as (for instance) Training or
    Validation, based on the metric name matching the given regexes. The default will
    use legend Validation for any metric name beginning 'val_', and legend 'Train' for all others
    :param plotted_against: label for the x-axis. The implied meaning of the index in the lists
    of metrics. Usually epoch.
    :param legend_loc: loc parameter for call to plt.legend() on each graph
    """
    legends_based_on=legends_based_on + (('—','.*'),)
    legends=[]
    for key in  what_to_plot:
        for metric_name in what_to_plot[key]:
            plt.plot( metrics[metric_name])
            legend= [ t[0] for t in legends_based_on if re.search(t[1], metric_name)  ][0]
            legends.append( legend )
        plt.title(key.capitalize() + " against " + plotted_against)
        plt.ylabel(key)
        plt.xlabel(plotted_against)
        plt.legend(legends, loc='upper left')
        plt.show(block=False)


if __name__ == '__main__':
    model:models.Model
    historyandtime:HistoryAndTime
    model, historyandtime=main()
    print('Timing and Metrics', historyandtime)
    what_to_plot = {
            'Accuracy': ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy'],
            'Loss'     : ['loss', 'val_loss']
        }
    plot_metrics(historyandtime[-1].history, what_to_plot)
