import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

def draw_svm(clf, x, y, x1_min, x1_max, x2_min, x2_max, target=None):
    """
    Draws the decision boundary of an svm.
    :param clf: sklearn.svm.SVC classifier
    :param x: data Nx2
    :param y: label N
    :param x1_min: minimum value of the x-axis of the plot
    :param x1_max: maximum value of the x-axis of the plot
    :param x2_min: minimum value of the y-axis of the plot
    :param x2_max: maximum value of the y-axis of the plot
    :param target: if target is set to path, the plot is saved to that path
    :return: None
    """
    y = y.astype(bool)
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 500),
                         np.linspace(x2_min, x2_max, 500))
    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    disc_z = z > 0
    plt.clf()
    plt.imshow(disc_z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.RdBu, alpha=.3)
    plt.contour(xx, yy, z, levels=[-1, 1], linewidths=2,
                linestyles='dashed', colors=['red', 'blue'], alpha=0.5)
    plt.contour(xx, yy, z, levels=[0], linewidths=2,
                linestyles='solid', colors='black', alpha=0.5)
    positives = x[y == 1]
    negatives = x[y == 0]
    plt.scatter(positives[:, 0], positives[:, 1], s=50, marker='o', color="none", edgecolor="black")
    plt.scatter(negatives[:, 0], negatives[:, 1], s=50, marker='s', color="none", edgecolor="black")
    sv_label = y[clf.support_]
    positive_sv = x[clf.support_][sv_label]
    negative_sv = x[clf.support_][~sv_label]
    plt.scatter(positive_sv[:, 0], positive_sv[:, 1], s=50, marker='o', color="white", edgecolor="black")
    plt.scatter(negative_sv[:, 0], negative_sv[:, 1], s=50, marker='s', color="white", edgecolor="black")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.gca().set_aspect('equal', adjustable='box')
    if target is None:
        plt.show()
    else:
        plt.savefig(target)

### TASK1
train_set = np.load('svm/task1/train_set.npy')
train_lbs = np.load('svm/task1/train_labels.npy')
Cs = [0.01, 0.1, 1, 10, 100]
for c in Cs:
    clf = SVC(kernel='linear', C = c)
    clf.fit(train_set,train_lbs)
    draw_svm(clf,train_set,train_lbs, 
            train_set[np.argmin(train_set,axis=0)[0]][0],
            train_set[np.argmax(train_set,axis=0)[0]][0],
            train_set[np.argmin(train_set,axis=0)[1]][1],
            train_set[np.argmax(train_set,axis=0)[1]][1],
    )

### TASK2
train_set = np.load('svm/task2/train_set.npy')
train_lbs = np.load('svm/task2/train_labels.npy')
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for k in kernels:
    clf = SVC(kernel=k, C = 1)
    clf.fit(train_set,train_lbs)
    draw_svm(clf,train_set,train_lbs, 
            train_set[np.argmin(train_set,axis=0)[0]][0],
            train_set[np.argmax(train_set,axis=0)[0]][0],
            train_set[np.argmin(train_set,axis=0)[1]][1],
            train_set[np.argmax(train_set,axis=0)[1]][1],
    )

### TASK3

train_set = np.load('svm/task3/train_set.npy')
train_lbs = np.load('svm/task3/train_labels.npy')
test_set = np.load('svm/task3/test_set.npy')
test_lbs = np.load('svm/task3/test_labels.npy')
length = np.shape(train_lbs)[0] # 1000
length_test = np.shape(test_lbs)[0]
train_set = train_set.reshape((1000,32*32))
test_set = test_set.reshape((1000,32*32))
val_set = train_set[800:]
val_lbs = train_lbs[800:]
train_set = train_set[:800]
train_lbs = train_lbs[:800]
cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.01, 0.1]
kernels = ['linear', 'rbf', 'poly']

for k in kernels:
    for c in cs:
        for g in gammas:
            clf = SVC(kernel=k, C = c, gamma=g)
            clf.fit(train_set,train_lbs)
            accuracy = 0
            for i in range(200):
                if clf.predict(val_set[i].reshape(1,-1)) == val_lbs[i]:
                    accuracy+=1
            if k == 'rbf':
                print("Kernel:", k, " C:", c, " Gamma:", g, " Validation Accuracy: %", accuracy/200*100)
            else:
                print("Kernel:", k, " C:", c, " Gamma:", g, " Validation Accuracy: %", accuracy/200*100)
                break
            

clf = SVC(kernel="rbf", C = 10, gamma=0.01)
clf.fit(train_set,train_lbs)
accuracy = 0
for i in range(200):
    if clf.predict(test_set[i].reshape(1,-1)) == test_lbs[i]:
        accuracy+=1
print("Test Kernel:rbf C:10 Gamma:0.01 Test Accuracy:%", accuracy/200*100)

### TASK4
train_set = np.load('svm/task4/train_set.npy')
train_lbs = np.load('svm/task4/train_labels.npy')
test_set = np.load('svm/task4/test_set.npy')
test_lbs = np.load('svm/task4/test_labels.npy')
lentrain = np.shape(train_lbs)[0]
lentest = np.shape(test_lbs)[0]

train_set = train_set.reshape((lentrain,32*32))
test_set = test_set.reshape((lentest,32*32))

clf = SVC(kernel='rbf', C = 1)
clf.fit(train_set,train_lbs)


accuracy = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(lentest):
    if clf.predict(test_set[i].reshape(1,-1)) == test_lbs[i]:
        accuracy+=1
        if test_lbs[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if test_lbs[i] == 1:
            fn += 1
        else:
            fp += 1

print("Positives in train set: ", np.sum(train_lbs))
print("Negatives in train set: ", np.count_nonzero(train_lbs == 0))
print("True positives on test set: ", tp)
print("True negatives on test set: ", tn)
print("False positives on test set: ", fp)
print("False negatives on test set: ", fn)
print("Accuracy on test set: %", accuracy/lentest*100)


#### oversampling
my_train_set = train_set
my_train_lbs = train_lbs

orj_negatives = my_train_set[my_train_lbs == 0]
len_orj = np.shape(orj_negatives)[0]
while np.count_nonzero(my_train_lbs == 0) != np.count_nonzero(my_train_lbs == 1):
    my_train_set = np.vstack((my_train_set,orj_negatives[(np.random.randint(0,len_orj))]))
    my_train_lbs = np.append(my_train_lbs,0)

clf = SVC(kernel='rbf', C = 1)
clf.fit(my_train_set,my_train_lbs)

accuracy = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(lentest):
    if clf.predict(test_set[i].reshape(1,-1)) == test_lbs[i]:
        accuracy+=1
        if test_lbs[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if test_lbs[i] == 1:
            fn += 1
        else:
            fp += 1
            
print("After Oversampling, Positives in train set: ", np.sum(my_train_lbs))
print("After Oversampling, Negatives in train set: ", np.count_nonzero(my_train_lbs == 0))
print("After Oversampling, True positives on test set: ", tp)
print("After Oversampling, True negatives on test set: ", tn)
print("After Oversampling, False positives on test set: ", fp)
print("After Oversampling, False negatives on test set: ", fn)
print("After Oversampling, Accuracy on test set: %", accuracy/lentest*100)


#### undersampling
my_train_set = train_set
my_train_lbs = train_lbs

while np.count_nonzero(my_train_lbs == 0) != np.count_nonzero(my_train_lbs == 1):
    random = np.random.randint(0,np.shape(my_train_lbs)[0])
    if my_train_lbs[random] == 1:
        my_train_lbs = np.delete(my_train_lbs,random,axis=0)
        my_train_set = np.delete(my_train_set,random,axis=0)

clf = SVC(kernel='rbf', C = 1)
clf.fit(my_train_set,my_train_lbs)

accuracy = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(lentest):
    if clf.predict(test_set[i].reshape(1,-1)) == test_lbs[i]:
        accuracy+=1
        if test_lbs[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if test_lbs[i] == 1:
            fn += 1
        else:
            fp += 1
            
print("After Undersampling, Positives in train set: ", np.sum(my_train_lbs))
print("After Undersampling, Negatives in train set: ", np.count_nonzero(my_train_lbs == 0))
print("After Undersampling, True positives on test set: ", tp)
print("After Undersampling, True negatives on test set: ", tn)
print("After Undersampling, False positives on test set: ", fp)
print("After Undersampling, False negatives on test set: ", fn)
print("After Undersampling, Accuracy on test set: %", accuracy/lentest*100)


####
clf = SVC(kernel='rbf', C = 1, class_weight='balanced')
clf.fit(train_set,train_lbs)
accuracy = 0
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(lentest):
    if clf.predict(test_set[i].reshape(1,-1)) == test_lbs[i]:
        accuracy+=1
        if test_lbs[i] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if test_lbs[i] == 1:
            fn += 1
        else:
            fp += 1

print("With 'balanced' Parameter, Positives in train set: ", np.sum(train_lbs))
print("With 'balanced' Parameter, Negatives in train set: ", np.count_nonzero(train_lbs == 0))
print("With 'balanced' Parameter, True positives on test set: ", tp)
print("With 'balanced' Parameter, True negatives on test set: ", tn)
print("With 'balanced' Parameter, False positives on test set: ", fp)
print("With 'balanced' Parameter, False negatives on test set: ", fn)
print("With 'balanced' Parameter, Accuracy on test set: %", accuracy/lentest*100)
