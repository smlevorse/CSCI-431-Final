import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout

import pickle
import os

from helper import getBinaryfer13Data, emotionToMatrix

def constructModel():
    model = Sequential()

    model.add(Conv2D(
        input_shape = (48, 48, 1),
        filters = 28,
        kernel_size = (5,5),
        strides = (1, 1)
        # padding = 'valid',
        # data_format = 'channels_last',
        # dilation_rate = (1, 1),
        # activation = 'relu',
        # use_bias = True,
        # kernel_initializer = 'random_uniform'
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Should result in 24x24x1 output

    model.add(Conv2D(
        filters = 8,
        kernel_size = (7, 7),
        # activation = 'relu',
        # use_bias = True,
        # kernel_initializer = 'random_uniform',
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

    model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

    return model

'''
Save time by not running the normalize function on each image each time we run the program
'''
def loadImages():
    allFilesExist = True
    allFilesExist = allFilesExist and os.path.exists('nomalizeTrainingImages') and os.path.exists('trainingLabels')
    allFilesExist = allFilesExist and os.path.exists('normalizedPublicValidationImages') and os.path.exists('publicValidationLabels')
    allFilesExist = allFilesExist and os.path.exists('normalizedPrivateValidationImages') and os.path.exists('privateValidationLabels')

    if allFilesExist:
        print("Normalized images found, unpickling...")
        with open('nomalizeTrainingImages', 'rb') as fi:
            imagesT = pickle.load(fi)
        with open('trainingLabels', 'rb') as fi:
            labelsT = pickle.load(fi)
        with open('normalizedPublicValidationImages', 'rb') as fi:
            imagesPubV = pickle.load(fi)
        with open('publicValidationLabels', 'rb') as fi:
            labelsPubV = pickle.load(fi)
        with open('normalizedPrivateValidationImages', 'rb') as fi:
            imagesPriV = pickle.load(fi)
        with open('privateValidationLabels', 'rb') as fi:
            labelsPriV = pickle.load(fi)
    else:
        print("No normalized images found, loading from CSV")
        imagesT, imagesPubV, imagesPriV, labelsT, labelsPubV, labelsPriV = getBinaryfer13Data('fer2013.csv')
        # Store normalized images in pickled files to save that ungodly normalizing time again
        with open('nomalizeTrainingImages', 'wb') as fi:
            pickle.dump(imagesT, fi)
        with open('trainingLabels', 'wb') as fi:
            pickle.dump(labelsT, fi)
        with open('normalizedPublicValidationImages', 'wb') as fi:
            pickle.dump(imagesPubV, fi)
        with open('publicValidationLabels', 'wb') as fi:
            pickle.dump(labelsPubV, fi)
        with open('normalizedPrivateValidationImages', 'wb') as fi:
            pickle.dump(imagesPriV, fi)
        with open('privateValidationLabels', 'wb') as fi:
            pickle.dump(labelsPriV, fi)

    return imagesT, imagesPubV, imagesPriV , labelsT, labelsPubV, labelsPriV

def main():
    print('loading images')
    imagesT, imagesPubV, imagesPriV , labelsT, labelsPubV, labelsPriV = loadImages()

    print('done loading')
    print('constructing model')
    model = constructModel()
    print('model constructed')
    print('training model')
    model.fit(imagesT, labelsT, epochs = 10, validation_data = (imagesPubV, labelsPubV))
    print('model trained')
    print('saving weights')
    model.save_weights('weights.h5')
    print('saved!')

if __name__ == "__main__":
    main()