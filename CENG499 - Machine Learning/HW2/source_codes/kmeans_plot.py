import numpy as np
from kmeans import assign_clusters
import matplotlib.pyplot as plt

colors = ['red', 'lime', 'fuchsia', 'dodgerblue', 'gold', 'darkcyan', 'darkgreen', 'mediumslateblue']

objs1 = np.load("outs/kmeans_objs1.npy")
plt.plot(range(1,11),objs1)
plt.show()
k1 = int(input("Best K for set1: "))
x, y = [], []
data = np.load("kmeans/dataset1.npy")
centers = np.load('outs/kmeans_centers1.npy',allow_pickle=True)
assignings = assign_clusters(data,centers[k1-1])
for c in range(k1):
    x, y = [], []
    for i,d in enumerate(data):
        if assignings[i] == c:
            x.append(data[i,0])
            y.append(data[i,1])
    plt.scatter(x.copy(), y.copy(), color=colors[c])
cx = [centers[k1-1][:,0]]
cy = [centers[k1-1][:,1]]
plt.scatter(cx, cy, color='black')
plt.show()


objs2 = np.load("outs/kmeans_objs2.npy")
plt.plot(range(1,11),objs2)
plt.show()
k2 = int(input("Best K for set2: "))
x, y = [], []
data = np.load("kmeans/dataset2.npy")
centers = np.load('outs/kmeans_centers2.npy',allow_pickle=True)
assignings = assign_clusters(data,centers[k2-1])
for c in range(k2):
    x, y = [], []
    for i,d in enumerate(data):
        if assignings[i] == c:
            x.append(data[i,0])
            y.append(data[i,1])
    plt.scatter(x.copy(), y.copy(), color=colors[c])

cx = [centers[k2-1][:,0]]
cy = [centers[k2-1][:,1]]
plt.scatter(cx, cy, color='black')
plt.show()

objs3 = np.load("outs/kmeans_objs3.npy")
plt.plot(range(1,11),objs3)
plt.show()
k3 = int(input("Best K for set3: "))
x, y = [], []
data = np.load("kmeans/dataset3.npy")
centers = np.load('outs/kmeans_centers3.npy',allow_pickle=True)
assignings = assign_clusters(data,centers[k3-1])
for c in range(k3):
    x, y = [], []
    for i,d in enumerate(data):
        if assignings[i] == c:
            x.append(data[i,0])
            y.append(data[i,1])
    plt.scatter(x.copy(), y.copy(), color=colors[c])
cx = [centers[k3-1][:,0]]
cy = [centers[k3-1][:,1]]
plt.scatter(cx, cy, color='black')
plt.show()

objs4 = np.load("outs/kmeans_objs4.npy")
plt.plot(range(1,11),objs4)
plt.show()
k4 = int(input("Best K for set4: "))
x, y = [], []
data = np.load("kmeans/dataset4.npy")
centers = np.load('outs/kmeans_centers4.npy',allow_pickle=True)
assignings = assign_clusters(data,centers[k4-1])
for c in range(k4):
    x, y = [], []
    for i,d in enumerate(data):
        if assignings[i] == c:
            x.append(data[i,0])
            y.append(data[i,1])
    plt.scatter(x.copy(), y.copy(), color=colors[c])
cx = [centers[k4-1][:,0]]
cy = [centers[k4-1][:,1]]
plt.scatter(cx, cy, color='black')
plt.show()