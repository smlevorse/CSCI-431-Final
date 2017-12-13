import tensorflow as tf
from keras.models import  Sequential
import numpy as np
from helper import normalizeImage, loadImages
from homework4 import constructModel

'''
Takes a 48x48 greyscale image as a numpy array and predicts a facial expression
'''
def predict(I, model):
    normalized = []
    normalized.append((normalizeImage(I.reshape((48,48))) / 255.0).reshape((48, 48, 1)))
    result = model.predict(np.array(normalized))[0]
    return np.argmax(result)

def main():
    print('loading model')
    model = constructModel()
    model.load_weights('weights.h5')
    print('loading images')
    imagesT, imagesPubV, imagesPriV, labelsT, labelsPubV, labelsPriV = loadImages()

    for i in range(10):
        print(str(predict(imagesPriV[i], model)) + ' ' + str(np.argmax(labelsPriV[i])))

if __name__ == "__main__":
    main()
