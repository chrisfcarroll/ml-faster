import time, operator, subprocess
from collections import namedtuple
from typing import Tuple, List, NamedTuple, Dict
from warnings import warn
from functools import reduce
from numpy import math
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, datasets, backend, losses
from tensorflow.python.keras.callbacks import History

MB=1024**2
GB=1024**3
class GPU_Ram: TeslaK80_1Proc= 12*GB; TeslaV100=16*GB; TeslaP100=16*GB; TeslaM60=16*GB; NoGPU=128*MB


def main(gpu_ram= GPU_Ram.TeslaK80_1Proc, epochs=20 ):

    mnist_4Dense_net= models.Sequential([
            layers.Reshape(target_shape= (28*28,),input_shape=(28, 28)),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='relu')]
            )
    mnist_4Dense_net.summary()
    ds_train,ds_val=create_mnist_datasets()
    best_batch_size=max_batch_size(gpu_ram,mnist_4Dense_net,default_max=64)
    run_result= train_and_eval(mnist_4Dense_net,
                               ds_train, ds_val,
                               epochs=epochs,
                               max_batch_size=best_batch_size)
    print(run_result)
    return (mnist_4Dense_net, run_result)


class EpochsSliceResult(NamedTuple):
    index:int
    num_epochs:int
    time_secs:int
    loss:float
    accuracy:float
    metrics: Dict[str, float]

class RunResults(List[EpochsSliceResult]):
    def num_epochs(self)->int       : return sum([e.num_epochs for e in self])
    def total_time(self)->float     : return sum([e.time_secs for e in self])
    def final_loss(self)->float     : return self[-1].loss
    def final_accuracy(self)->float : return self[-1].accuracy
    def final_metrics(self)->Dict[str, float] : return self[-1].metrics



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
    for your model and inputs
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
    tensors_size= all_inputs + outputs
    num_weights=sum(
            [ a.shape.num_elements()
              for a in model.trainable_weights + model.non_trainable_weights ])
    num_gradients=num_weights
    num_scalars=tensors_size + num_weights + num_gradients
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
                   epochs=40, eval_every_n_epochs=10,
                   save_to_path='TrainedModel',
                   max_batch_size=32):
    print('compiling...')
    model.compile(
            optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy()])
    try: subprocess.run('nvidia-smi')
    except Exception: pass

    run_result= RunResults()

    print('training...')
    for i in range(1, 1+epochs//eval_every_n_epochs):
        start=time.time()
        epoch=1+(i-1)*eval_every_n_epochs
        batch_size=decay_pow2(epoch - 1, epochs, max_batch_size)
        ds_batch=ds_train.batch(batch_size)
        epochs_descr='Epochs {}-{} batch size {} '.format(epoch,epoch+eval_every_n_epochs-1, batch_size)
        print(epochs_descr)
        history=model.fit(ds_batch,
                 epochs=eval_every_n_epochs,
                 validation_data=ds_val.batch(batch_size),
                 validation_freq=eval_every_n_epochs)
        tf.keras.models.save_model(model,save_to_path)
        val_loss,val_accuracy= model.evaluate(ds_val.batch(batch_size))
        epochs_slice_time = time.time() - start
        print(epochs_descr,'took {:.1f}sec. Validation loss= {:.2f}, Validation Accuracy= {}%' \
              .format(epochs_slice_time, val_loss, int(val_accuracy * 100)))
        epoch_result=EpochsSliceResult(i,
                                       eval_every_n_epochs,
                                       epochs_slice_time,
                                       val_loss,
                                       val_accuracy,
                                       history.history)
        run_result.append(epoch_result)
        return (model,run_result)

def warnif(condition:bool, message:str, **kwargs):
    if condition: warn(message,**kwargs)

if __name__ == '__main__':
    model,run_result=main()
