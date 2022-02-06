import numpy as np
import knn

test_labels = np.load("knn/test_labels.npy")
test_set = np.load("knn/test_set.npy")
train_labels = np.load("knn/train_labels.npy")
train_set = np.load("knn/train_set.npy")

avg_accs_l1 = []
avg_accs_l2 = []
for k in range(1,180):
    avg_accs_l1.append(knn.cross_validation(train_set,train_labels,10,k,'L1'))
    avg_accs_l2.append(knn.cross_validation(train_set,train_labels,10,k,'L2'))

np.save('outs/knn_avg_accs_l1.npy',np.array(avg_accs_l1))
np.save('outs/knn_avg_accs_l2.npy',np.array(avg_accs_l2))