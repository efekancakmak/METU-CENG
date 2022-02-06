import numpy as np
import matplotlib.pyplot as plt
from knn import calculate_distances, majority_voting, knn, split_train_and_validation, cross_validation

x = range(1,180)
avg_accs_l1 = np.load('outs/knn_avg_accs_l1.npy')
avg_accs_l2 = np.load('outs/knn_avg_accs_l2.npy')

plt.plot(x,avg_accs_l1)
plt.show()
best_k_l1 = int(input("Best K for L1: "))
print("Test Accuracy for L1: ",knn(np.load('knn/train_set.npy'),np.load('knn/train_labels.npy'),np.load('knn/test_set.npy'),np.load('knn/test_labels.npy'),best_k_l1,'L1'))


plt.plot(x,avg_accs_l2)
plt.show()
best_k_l2 = int(input("Best K for L2: "))
print("Test Accuracy for L2: ",knn(np.load('knn/train_set.npy'),np.load('knn/train_labels.npy'),np.load('knn/test_set.npy'),np.load('knn/test_labels.npy'),best_k_l2,'L2'))


