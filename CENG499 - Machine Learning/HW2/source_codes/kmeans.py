import numpy as np


def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    i = 0
    assign = []
    for d in data:
        min = 99**9
        index = -1
        for i,c in enumerate(cluster_centers):
            temp = np.sqrt(np.sum((d-c)**2))
            if temp < min:
                index = i
                min = temp
        assign.append(index)
    return np.array(assign)


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    clusters = []
    for i in range(k):
        clusters.append([])
    for i,d in enumerate(data):
        clusters[assignments[i]].append(d)
    i = 0
    while i < len(clusters):
        if np.shape(clusters[i])[0]==0:
            clusters[i] = cluster_centers[i]
        else:
            clusters[i] = np.sum(clusters[i],axis=0)/(np.shape(clusters[i])[0])
        i += 1
    return np.array(clusters)

def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    centers = initial_cluster_centers
    k = np.shape(initial_cluster_centers)[0]
    while True:
        assigned = assign_clusters(data,centers)
        temp = centers
        centers = calculate_cluster_centers(data,assigned,centers,k)
        if np.all(np.abs(centers - temp) < 10 ** -5):
            break
    obj = 0.0
    for i,d in enumerate(data):
        obj += (np.sum((d-centers[assigned[i]])**2))
    return centers,obj/2