import numpy as np
import matplotlib.pyplot as plt


def ecludian(x, y):
    tempx = (x[0] - x[1]) ** 2
    tempy = (y[0] - y[1]) ** 2
    result = (tempx + tempy) ** 0.5
    return result

def kmean(data, center, n):
    label = np.zeros(len(data))
    clus0 = []
    clus1 = []
    print(center)
    for epho in range(n):
        for i in range(len(data)):
            temp0 = ecludian(data[i], center[0])
            temp1 = ecludian(data[i], center[1])
            label[i] = np.argmin([temp0, temp1])
            print(i, temp0, temp1)
            if label[i] == 1:
                clus1.append(data[i])
            else:
                clus0.append(data[i])
        center[0] = np.average(np.array(clus0), axis=0)
        center[1] = np.average(np.array(clus1), axis=0)
        print(center)


data = np.array([[7,5],[5,7],[7,7],[3,3],[4,6],[1,4],[0,0],[2,2],[8,7],[6,8]])
center = np.array([[4,6],[5,5]], dtype=float)

print(kmean(data, center, 1))
