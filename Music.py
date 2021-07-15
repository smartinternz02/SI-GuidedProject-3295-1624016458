import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

import os
import pickle
import operator


def distance(instance1, instance2, k):
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]

    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbours(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours


def nearestClass(neighbours):
    classVote = {}

    for x in range(len(neighbours)):
        response = neighbours[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)

    return sorter[0][0]


directory = "C:/Users/HP/PycharmProjects/flaskProject/MusicGenre/"
f = open("my.dat", "wb")

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory + folder):
        (rate, sig) = wav.read(directory + folder + "/" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)
f.close()

dataset = []


def loadDataSet(filename):
    with open("C:/Users/HP/PycharmProjects/flaskProject/my.dat", 'rb') as f1:
        while True:
            try:
                dataset.append(pickle.load(f1))
            except EOFError:
                f1.close()
                break


loadDataSet("C:/Users/HP/PycharmProjects/flaskProject/my.dat")

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(dataset, test_size=0.2)

leng = len(x_test)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbours(x_train, x_test[x], 5)))


def getAccuracy(testSet):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (1.0 * correct) / len(testSet)


accuracy1 = getAccuracy(x_test)
print(accuracy1)
