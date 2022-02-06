from sklearn import tree
import numpy as np
import graphviz 
from matplotlib import pyplot as plt

train_set = np.load( 'dt/train_set.npy')
train_labels = np.load( 'dt/train_labels.npy')
test_set = np.load( 'dt/test_set.npy')
test_labels = np.load( 'dt/test_labels.npy')


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_set, train_labels)
length = np.shape(test_set)[0]
accuracy = 0

fig, ax = plt.subplots(figsize=(8, 8))
#fig = plt.figure(figsize=(10, 10))
_ = tree.plot_tree(clf, 
                   feature_names=['Attribute0','Attribute1','Attribute2','Attribute3'],  
                   class_names=["Class0","Class1","Class2"],
                   filled=True,
                   ax=ax)
#r = tree.export_text(clf, feature_names=["A0","A1","A2","A3"])
plt.show()
fig.savefig("decistion_tree.png")


dot_data = tree.export_graphviz(clf, out_file="out.dot") 
print(dot_data)
graph = graphviz.Source(dot_data,format="png") 
print(graph)
graph.render("iris") 