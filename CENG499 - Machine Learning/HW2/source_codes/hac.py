import numpy as np


def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    min = 99**9
    for x in c1:
        for y in c2:
            distance = np.sqrt(np.sum((x-y)**2))
            if distance < min:
                min = distance
    return min

def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    max = 0
    for x in c1:
        for y in c2:
            distance = np.sqrt(np.sum((x-y)**2))
            if distance > max:
                max = distance
    return max

def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = np.array([])
    for x in c1:
        for y in c2:
            distances = np.append(distances,np.sqrt(np.sum((x-y)**2)))
    return np.sum(np.array(distances))/np.shape(distances)[0]

def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    return np.sqrt(np.sum((np.sum(c1,axis=0)/np.shape(c1)[0] - np.sum(c2,axis=0)/np.shape(c2)[0])**2))
    
def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    clusters = [[i,np.array([d])] for i,d in enumerate(data)]
    number_of_clusters = len(clusters)
    while number_of_clusters > stop_length:
        min = 99**9
        which = (-1,-1)
        for x in clusters:
            for y in clusters:
                if x==y:
                    break
                distance = criterion(x[1],y[1])
                if distance < min:
                    min = distance
                    which = (x[0],y[0])
        new_cluster = []
        i = 0
        while i < len(clusters):
            if clusters[i][0] == which[0] or clusters[i][0] == which[1]:
                new_cluster.append(clusters[i])
                clusters.pop(i)
                i-=1
            i+=1
        new_cluster = [new_cluster[0][0], np.concatenate((new_cluster[0][1],new_cluster[1][1]),axis=0)]
        clusters.append(new_cluster)
        number_of_clusters -= 1 
    clusters = [i[1] for i in clusters]
    return np.array(clusters)