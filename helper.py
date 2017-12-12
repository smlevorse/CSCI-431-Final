import numpy as np
import cv2
from sklearn.preprocessing import normalize

def getBinaryfer13Data(filename):
    labelsTrain = []
    labelsPubVal = []
    labelsPriVal = []
    imagesTrain = []
    imagesPubVal = []
    imagesPriVal = []
    first = True
    count = 0
    for line in open(filename, 'rU'):
        if first:
            first = False
        else:
            if count % 1000 == 0:
                print("Processing image " + str(count))
            row = line.split(',')
            y = emotionToMatrix(int(row[0]))
            image = row[1].split(' ')
            reshaped = normalizeImage(np.reshape(image, (48, 48, 1)).astype(float)) / 255
            image_type = row[2]
            if image_type.find('Training') >= 0:
                labelsTrain.append(y)
                imagesTrain.append(reshaped)
            elif image_type.find('Public') >= 0:
                labelsPubVal.append(y)
                imagesPubVal.append(reshaped)
            else:
                labelsPriVal.append(y)
                imagesPriVal.append(reshaped)
        count = count + 1
    return np.array(imagesTrain), np.array(imagesPubVal), np.array(imagesPriVal), np.array(labelsTrain), np.array(labelsPubVal), np.array(labelsPriVal)

def emotionToMatrix(y):
    output = np.zeros(7)
    output[y] = 1
    return output

'''
Gets the standard deviation in a 7x7 neighborhood around the specified pixel in the provided image
Code created with help from:
https://stackoverflow.com/questions/44906530/how-to-find-the-gaussian-weighted-average-and-standard-deviation-of-a-structural
'''
def StdDev(I, meanPoint, point):
    # Check for edges of image
    ystart = point[1] - 3 if 0 < point[1] - 3 < I.shape[0] else 0
    yend = point[1] + 3 + 1 if 0 < point[1] + 3 + 1 < I.shape[0] else I.shape[0] - 1

    xstart = point[0] - 3 if 0 < point[0] - 3 < I.shape[1] else 0
    xend = point[0] + 3 + 1 if 0 < point[0] + 3 + 1 < I.shape[1] else I.shape[1] - 1

    patch = (I[ystart:yend, xstart:xend] - meanPoint) ** 2
    total = np.sum(patch)
    n = patch.size

    return 1 if total == 0 or n == 0 else np.sqrt(total / float(n))

'''
Nomalizes an image by performing x' = (x-gaussian_average)/(neighborhood_std)
Based on "Facial expression recognition with Convolutional Neural Networks:
Coping with few data and the training sample order" by Lopes, Aguiar, De Souza, and 
Oliveira-Santos. 
Code created with help from:
https://stackoverflow.com/questions/44906530/how-to-find-the-gaussian-weighted-average-and-standard-deviation-of-a-structural
'''
def normalizeImage(I):
    # Run a gaussian blur on the entire image with a kernel of size 7
    blur = cv2.GaussianBlur(I, (7, 7), 0, 0).astype(np.float)
    normalized = np.ones(I.shape, dtype=np.float) * 127

    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            # For each pixel, get x, gaussian_average, and stddev, to calculate x prime
            x = I[j, i]
            gauss = blur[j, i]
            std = StdDev(I, gauss, [i, j])

            xp = 127  # default x prime in the middle of the range
            if std > 0:
                xp = (x - gauss) / float(std)

            xp = np.clip((xp * 127 / float(2.0)) + 127, 0, 255)  #clip value to byte sized
            normalized[j, i] = xp
    return normalized
