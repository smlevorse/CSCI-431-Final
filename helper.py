import numpy as np

def getBinaryfer13Data(filename):
    labelsTrain = []
    labelsPubVal = []
    labelsPriVal = []
    imagesTrain = []
    imagesPubVal = []
    imagesPriVal = []
    first = True
    for line in open(filename, 'rU'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = emotionToMatrix(int(row[0]))
            image = row[1].split(' ')
            reshaped = np.reshape(image, (48, 48, 1)).astype(float) / 255.0
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
    return np.array(imagesTrain), np.array(imagesPubVal), np.array(imagesPriVal), np.array(labelsTrain), np.array(labelsPubVal), np.array(labelsPriVal)

def emotionToMatrix(y):
    output = np.zeros(7)
    output[y] = 1
    return output
