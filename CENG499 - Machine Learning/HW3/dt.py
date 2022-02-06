import numpy as np

def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    total = sum(bucket)
    entropy = 0.0
    for i in bucket:
        if i!=0:
            entropy -= (i/total)*np.log2(i/total)
    return entropy

def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    Es = entropy(parent_bucket)
    El = entropy(left_bucket)
    Er = entropy(right_bucket)
    left = sum(left_bucket)
    right = sum(right_bucket)
    return Es - (left/(left+right)*El + right/(left+right)*Er)

def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    total = sum(bucket)
    res = 1
    for i in bucket:
        res -= (i/total)**2
    return res

def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    l = gini(left_bucket)
    r = gini(right_bucket)
    left = sum(left_bucket)
    right = sum(right_bucket)
    return left/(left+right)*l + right/(left+right)*r

def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """ 
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    column = data[:,attr_index]
    column_labels = []
    for i in range(len(labels)):
        column_labels.append([i,column[i],labels[i]])

    column_labels.sort(key=lambda x: x[1])
    column_labels = np.array(column_labels)

    split_vals = []
    heuristics = []

    for i in range(1,len(labels)):
        split_vals.append( (column_labels[i-1][1]+column_labels[i][1])/2 )
        left = column_labels[:i]
        right = column_labels[i:]
        l, r, p = [], [], []
        for j in range(num_classes):
            p.append(np.count_nonzero(column_labels[:,2]==j))
            l.append(np.count_nonzero(left[:,2]==j))
            r.append(np.count_nonzero(right[:,2]==j))
        if heuristic_name == 'info_gain':
            heuristics.append(info_gain(p,l,r))
        elif heuristic_name == 'avg_gini_index':
            heuristics.append(avg_gini_index(l,r))
        else:
            print("Wrong heuristic..")
            exit()
    split_vals = np.array(split_vals)
    heuristics = np.array(heuristics)
    return np.column_stack((split_vals, heuristics))

def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    length_l = len(left_bucket)
    parent = (np.array(left_bucket) + np.array(right_bucket))
    chi = 0
    for i in range(length_l):
        expected1 = parent[i]/np.sum(parent)*np.sum(np.array(left_bucket))
        expected2 = parent[i]/np.sum(parent)*np.sum(np.array(right_bucket))
        actual1 = left_bucket[i]
        actual2 = right_bucket[i]
        if expected1 != 0:
            chi += (actual1-expected1)**2/expected1 
        if expected2 != 0:
            chi += (actual2-expected2)**2/expected2 

    degree = np.count_nonzero((np.array(left_bucket) + np.array(right_bucket)) != 0 ) - 1
    return chi,degree