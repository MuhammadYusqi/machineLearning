import numpy as np
import csv
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def loadCSV(fileName):
    data = []
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            data.append(row)
    return np.array(data[1:], dtype=float)


def linearRegression(x, y):
    xt = np.transpose(x)
    xinv = np.linalg.inv(np.dot(xt, x))
    temp = np.dot(xinv, xt)
    w = np.dot(temp, y)
    return w

def predict(w, data):
    temp = w[1:] * data
    temp1 = np.concatenate((w[0],temp), axis=None)
    return sum(temp1)

def predicts(w, datas):
    jumlahBaris = datas.shape[0]
    result = np.zeros(jumlahBaris)
    for i in range(jumlahBaris):
        result[i] = predict(w, datas[i])
    return result

def R2(target, predict):
    sstot = sum((target - np.average(target)) ** 2)
    ssres = sum((target - predict) ** 2)
    result = 1 - (ssres/sstot)
    return(result)


dataset = datasets.load_diabetes()
dataTrain = dataset.data
target = dataset.target
ones = np.ones((dataTrain.shape[0], 1))
# print(ones.shape)
# print(dataTrain.shape)
dataTrain1 = np.concatenate((ones, dataTrain), axis=1)
# manual
w = linearRegression(dataTrain1, target)
predict = predicts(w, dataTrain)

# scikitLearn
reg = LinearRegression().fit(dataTrain, target)

print(R2(target, predict))
print(reg.score(dataTrain, target))
print(r2_score(target, predict))
