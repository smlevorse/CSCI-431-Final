import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D


from helper import getBinaryfer13Data

def constructModel():
    model = Sequential()
    model.add(Convolution2D(
        input_shape = (48, 48, 1),
        filters = 16,
        kernel_size = 5,
        strides = (1, 1),
        padding = 'valid',
        data_format = 'channels_last',
        dilation_rate = (1, 1),
        activation = 'relu',
        use_bias = True,
        kernel_initializer = 'random_uniform',
        bias_initializer = 'zeros',
        kernel_regulizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None
    ))
    model.add(MaxPool2D(pool_size = (2, 2)))
    # Should result in 24x24x1 output

    model.add(Convolution2D(
        filters = 16,
        kernel_size = 5,
        activation = 'relu',
        use_bias = True,
        kernel_initializer = 'random_uniform',
        bias_initializer = 'zeros',
    ))
    model.add(MaxPool2D(pool_size = (2, 2)))
    # Should result in 12x12x1 output

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    imagesT, imagesV, labelsT, labelsV = getBinaryfer13Data('fer2013.csv')


