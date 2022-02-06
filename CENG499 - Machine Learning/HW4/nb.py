import numpy as np
import math

def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    Set = set()
    for sentence in data:
        for word in sentence:
            Set.add(word)
    return Set
    

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    length = len(train_labels)
    pi = {}
    for cl in train_labels:
        pi[cl] = 1 if cl not in pi else pi[cl] + 1
    for cl in pi:
        pi[cl] /= length
    return pi
    
    
def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    #         { class1: {word: estimated prob, word2: estimated prob}, class2: ...}
    theta = {}
    class_amounts = {}
    length = len(train_labels)
    for i in range(length):
        if train_labels[i] not in theta:
            theta[train_labels[i]] = {}
            class_amounts[train_labels[i]] = 0
            for v in vocab:
                theta[train_labels[i]][v] = 1
                class_amounts[train_labels[i]] += 1
        for word in train_data[i]:
            theta[train_labels[i]][word] += 1
            class_amounts[train_labels[i]] += 1
    for cl in theta:
        for w in theta[cl]:
            theta[cl][w] /= (class_amounts[cl])
    return theta


def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    #  [ ex1, ex2, ... ]
    #  [ [(score11,class1), (score12,class2), ...], [(score21,class1),(score22,class2), ...], ...]
    scores, classes = [], []
    for p in pi:
        classes.append(p)
    for t in test_data:
        scores.append([])
        for c in classes:
            words = filter(lambda w: w in vocab, t)
            words = [np.log(theta[c][w]) for w in words]
            scores[-1].append((np.log(pi[c]) + sum(words),c))
    return scores