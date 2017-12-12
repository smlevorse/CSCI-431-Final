import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Activation, Dense


from helper import getBinaryfer13Data, emotionToMatrix

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
        kernel_initializer = 'random_uniform'
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

    model.add(Flatten())
    model.add(Dense(7))
    model.add(Activation('relu'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

    return model

def main():
    print('loading images')
    imagesT, imagesV, labelsT, labelsV = getBinaryfer13Data('fer2013.csv')

    print(imagesT.shape)
    print(labelsT.shape)

    print('done loading')
    print('constructing model')
    model = constructModel()
    print('model constructed')
    print('training model')
    model.fit(imagesT, labelsT, epochs = 50, validation_data = (imagesV, labelsV))
    print('model trained')
    print('saving weights')
    model.save_weights('weights.h5')
    print('saved!')


if __name__ == "__main__":
    main()