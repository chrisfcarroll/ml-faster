import time
from typing import Tuple
from warnings import warn

import numpy as np
from numpy import ndarray, math
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models, optimizers, metrics, datasets, backend, losses
import os
import operator
from functools import reduce
MB=1024**2
GB=1024**3
class GPU_Ram: TeslaK80_1Proc= 12*GB; TeslaK80_2Proc=24*GB; TeslaV100=16*GB; TeslaM60=16*GB; NoGPU=128*MB

def main(gpu_ram= GPU_Ram.NoGPU):
    mnist_4Dense_net= models.Sequential([
            layers.Reshape(target_shape= (28*28,),input_shape=(28, 28)),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='relu')]
            )
    mnist_4Dense_net.summary()
    best_batch_size=max_batch_size(gpu_ram,mnist_4Dense_net)
    print('calculated batch size for {} GB RAM = {}'.format(gpu_ram/GB, best_batch_size))
    ds_train,ds_val=create_mnist_datasets(best_batch_size)
    train_and_eval(mnist_4Dense_net,ds_train,ds_val)



@tf.function
def mnist_to_float32_int(x,y)-> Tuple[tf.Tensor,tf.Tensor]:
    return tf.cast(x,tf.float32)/255.0, tf.cast(y,tf.int32)


def create_mnist_datasets(batch_size:int=64)->Tuple[tf.data.Dataset, tf.data.Dataset]:

    (x_train, y_train), (x_val, y_val)= datasets.mnist.load_data()
    ds_train=tf.data.Dataset.from_tensor_slices( (x_train,y_train)) \
                .map(mnist_to_float32_int)\
                .shuffle(batch_size).batch(batch_size)
    ds_val=tf.data.Dataset.from_tensor_slices((x_val,y_val)) \
                .map(mnist_to_float32_int)\
                .shuffle(batch_size).batch(batch_size)
    return ds_train,ds_val


def max_batch_size(gpu_ram_bytes:int,
                   model:models.Model, model_dtype_bytes:int=4,
                   usable=0.95, verbose=True)->int:
    assert 0 < gpu_ram_bytes, 'required: 0 < gpu_ram_bytes, you said %r' % gpu_ram_bytes
    assert 0 < usable, 'required: 0 < usable, you said %r' % usable
    assert 0 < model_dtype_bytes, 'required: 0< model_dtype_width, you said %r' % model_dtype_bytes
    assert model and model.layers, 'model.layers must not be None or empty'
    warnif(usable>1, "You've set usable GPU memory usage to more than 100%")
    model_total_tensor_size=sum(list(
            map(lambda shape : prod([dim if dim else 1 for dim in shape]),
                [ l.input_shape for l in model.layers ])))
    model_num_weights=sum(
            [ a.shape.num_elements()
              for a in model.trainable_weights + model.non_trainable_weights ])
    max_size= int(usable * gpu_ram_bytes / model_dtype_bytes / (model_total_tensor_size + model_num_weights))
    best_size= 2**int(math.floor(math.log(max_size, 2)))
    if verbose:
        print('Calculated total_tensor_size={}, num_weights={}, max batch size for {}GB is {}, best size is {}'\
              .format(model_total_tensor_size,model_num_weights,gpu_ram_bytes/GB,max_size,best_size))
    return best_size


def train_and_eval(
        model:models.Model,
        ds_train:tf.data.Dataset, ds_val:tf.data.Dataset,
        epochs=100, eval_every_n_epochs=10,
        save_to_path='TrainedModel'):
    print('compiling...')
    model.compile(
            optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.SparseCategoricalAccuracy()])
    print('training')
    for i in range(1,epochs//eval_every_n_epochs + 1):
        start=time.time()
        model.fit(ds_train,epochs=eval_every_n_epochs)
        tf.keras.models.save_model(model,save_to_path)
        result=model.evaluate(ds_val)
        print(result)
        val_loss,val_accuracy=result
        print('Epoch {:3} took {:3}: Validation loss={:.3f}, Validation accuracy={:3}%'\
                .format(i*eval_every_n_epochs,
                        time.time()-start,
                        val_loss,int(val_accuracy*100)))


def prod(iterable): return reduce(operator.mul, iterable)

def warnif(condition:bool, message:str, **kwargs):
    if condition: warn(message,**kwargs)

if __name__ == '__main__':
    main()