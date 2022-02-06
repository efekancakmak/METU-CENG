import numpy as np
from kmeans import kmeans
import os

def init_clusters(k,l):
    # k: amount of clusters
    # l: vector size of an instance
    return np.random.rand(k,l)

set1 = np.load("kmeans/dataset1.npy")
set2 = np.load("kmeans/dataset2.npy")
set3 = np.load("kmeans/dataset3.npy")
set4 = np.load("kmeans/dataset4.npy")

centers1 = []
centers2 = []
centers3 = []
centers4 = []
objs1 = []
objs2 = []
objs3 = []
objs4 = []
for i in range(1,11):
    min1 = 99**9
    min2 = 99**9
    min3 = 99**9
    min4 = 99**9

    for r in range(10):
        # TRY WITH 10 DIFFERENT INITIAL CLUSTER CONF.
        # AND SELECT ONES WHO HAVE MINIMUM OBJECTIVE FUNC.

        temp = kmeans(set1,init_clusters(i,2))
        if min1 > temp[1]:
            res1 = temp
            min1 = temp[1]

        temp = kmeans(set2,init_clusters(i,2))
        if min2 > temp[1]:
            res2 = temp
            min2 = temp[1]

        temp = kmeans(set3,init_clusters(i,2))
        if min3 > temp[1]:
            res3 = temp
            min3 = temp[1]
        temp = kmeans(set4,init_clusters(i,2))
        if min4 > temp[1]:
            res4 = temp
            min4 = temp[1]

    centers1.append(res1[0])
    objs1.append(res1[1])
    centers2.append(res2[0])
    objs2.append(res2[1])
    centers3.append(res3[0])
    objs3.append(res3[1])
    centers4.append(res4[0])
    objs4.append(res4[1])

np.save('outs/kmeans_centers1.npy',np.array(centers1))
np.save('outs/kmeans_centers2.npy',np.array(centers2))
np.save('outs/kmeans_centers3.npy',np.array(centers3))
np.save('outs/kmeans_centers4.npy',np.array(centers4))

np.save('outs/kmeans_objs1.npy',np.array(objs1))
np.save('outs/kmeans_objs2.npy',np.array(objs2))
np.save('outs/kmeans_objs3.npy',np.array(objs3))
np.save('outs/kmeans_objs4.npy',np.array(objs4))
