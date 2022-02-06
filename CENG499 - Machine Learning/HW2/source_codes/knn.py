import numpy as np


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    result = []
    for instant in train_data:
        if distance_metric == 'L1':
            result.append(np.sum(np.abs(test_instance-instant)))
        elif distance_metric == 'L2':
            result.append(np.sqrt(np.sum((test_instance-instant)**2)))
        else:
            print("Wrong Distance Metric!")
    return np.array(result)

def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    votes = {}
    dl = []
    """
    purpose here is to generate list like
    [ [<distance> , <label> ], ... ]
    """
    for i in range(np.shape(distances)[0]):
        dl.append([distances[i],labels[i]])
    dl.sort(key=lambda x: x[0])
    dl = dl[:k]
    for i in dl:
        votes[i[1]] = 0
    for i in dl:
        votes[i[1]] += 1
    # All kNNs are from the same class
    if len(votes)==1:
        return list(votes.keys())[0]
    votes = [[i,votes[i]] for i in votes]
    votes.sort(key=lambda x: x[0],reverse=True)
    votes.sort(key=lambda x: x[1])
    return votes[-1][0]

# def calculate_distances(train_data, test_instance, distance_metric):
# def majority_voting(distances, labels, k):
def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    votes = []
    for i in test_data:
        dists = calculate_distances(train_data, i, distance_metric)
        votes.append(majority_voting(dists,train_labels,k))
    length = len(votes)
    return np.sum(np.array(votes) == test_labels)/length


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    length = len(whole_train_data)
    split_len = int(length/k_fold)
    #"""
    train_data = np.concatenate((whole_train_data[:split_len*validation_index], whole_train_data[split_len*(validation_index+1):]))
    val_data = whole_train_data[split_len*validation_index:split_len*(validation_index+1)]
    train_label = np.concatenate((whole_train_labels[:split_len*validation_index], whole_train_labels[split_len*(validation_index+1):]))
    val_label = whole_train_labels[split_len*validation_index:split_len*(validation_index+1)]
    #"""
    return train_data,train_label,val_data,val_label

def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    cum_acc = 0
    for i in range(k_fold):
        train_data, train_label, val_data, val_label = split_train_and_validation(whole_train_data,whole_train_labels,i,k_fold)
        cum_acc += knn(train_data,train_label,val_data,val_label,k,distance_metric)
    return cum_acc/k_fold