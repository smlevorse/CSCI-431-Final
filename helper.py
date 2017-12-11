import numpy as np

def getBinaryfer13Data(filename):
    labelsTrain = []
    labelsValid = []
    imagesTrain = []
    imagesValid = []
    first = True
    for line in open(filename, 'rU'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            image = row[1]
            type = row[2]
            if(type == 'Training'):
                labelsTrain.append(y)
                imagesTrain.append(np.reshape(image, (48, 48)))
            else:
                labelsValid.append(y)
                imagesValid.append(np.reshape(image, (48, 48)))
    return np.array(imagesTrain) / 255.0, np.array(imagesValid) / 255.0, np.array(labelsTrain), np.array(labelsValid)