from __future__ import division, print_function
import numpy as np
from flask import Flask, request, render_template
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pickle
import operator

app = Flask(__name__)

dataset = []


def loadDataSet(filename):
    with open("C:/Users/HP/PycharmProjects/flaskProject/my.dat", 'rb') as f1:
        while True:
            try:
                dataset.append(pickle.load(f1))
            except EOFError:
                f1.close()
                break


loadDataSet("my.dat")


def distance(instance1, instance2, k):
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]

    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    print("distance is",distance)
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
    print("neighbours is",neighbours)
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
    print("sorter is", sorter)
    return sorter[0][0]


@app.route('/', methods=['GET'])
def index():
    return render_template('music.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = "C:/Users/HP/PycharmProjects/flaskProject/uploads/"+f.filename
        f.save(file_path)
        print(file_path)
        i = 1
        results = {1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'}
        (rate, sig) = wav.read(file_path)
        print(rate, sig)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, 0)
        pred = nearestClass(getNeighbours(dataset, feature, 8))
        print("predected genre = ", pred, "class = ", results[pred])
        res = "This song is classified as a "+str(results[pred])
        return render_template('music.html', pre=res)


if __name__ == '__main__':
    app.run(threaded=False)
