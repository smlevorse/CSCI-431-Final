import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout

from helper import loadImages

def constructModel():
    model = Sequential()

    model.add(Conv2D(
        input_shape = (48, 48, 1),
        filters = 14,
        kernel_size = (5,5),
        strides = (1, 1),
        # padding = 'valid',
        # data_format = 'channels_last',
        # dilation_rate = (1, 1),
        # activation = 'relu',
        # use_bias = True,
        kernel_initializer = 'random_uniform'
    ))
    model.add(Conv2D(
        input_shape=(48, 48, 1),
        filters=14,
        kernel_size=(5, 5),
        strides=(1, 1),
        # padding = 'valid',
        # data_format = 'channels_last',
        # dilation_rate = (1, 1),
        # activation = 'relu',
        # use_bias = True,
        kernel_initializer='random_uniform'
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Should result in 24x24x1 output

    model.add(Conv2D(
        filters = 8,
        kernel_size = (7, 7),
        # activation = 'relu',
        # use_bias = True,
        kernel_initializer = 'random_uniform',
        # bias_initializer = 'zeros',
    ))
    model.add(Conv2D(
        filters=8,
        kernel_size=(7, 7),
        # activation = 'relu',
        # use_bias = True,
        kernel_initializer='random_uniform',
        # bias_initializer = 'zeros',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Should result in 12x12x1 output

    model.add(Flatten())
    model.add(Dense(144))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    return model

def main():
    print('loading images')
    imagesT, imagesPubV, imagesPriV , labelsT, labelsPubV, labelsPriV = loadImages()

    print('done loading')
    print('constructing model')
    model = constructModel()
    print('model constructed')
    print('training model')
    history = model.fit(imagesT, labelsT, epochs = 20, validation_data = (imagesPubV, labelsPubV))
    print('model trained')
    print('saving weights')
    model.save_weights('weights.h5')
    print('saved!')

    # Generate Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Public Validation'])
    plt.savefig('Accuracy.png')

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Public Validation'])
    plt.savefig('Loss.png')


if __name__ == "__main__":
    main()