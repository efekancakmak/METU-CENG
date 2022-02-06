import dt
import numpy as np

"""
ID3(Examples, Target, Attributes)
    Create a root node
    If all Examples have the same Target value, give the root that label
    Else if Attributes is empty label, the root according to the most common value
    Else begin
        Calculate the information gain for each attribute, according to the average entropy formula
        Select the attribute, A, with the lowest average entropy (highest information gain) and make this the attribute tested at the root
        For each possible value, v, of this attribute
            Add a new branch below the root, corresponding to A = v
            Let Examples(v) be those examples with A = v
            If Examples(v) is empty, make the new branch a leaf node labelled with the most common value among Examples
            Else let the new branch be the tree created by ID3(Examples(v), Target, Attributes - {A}) 
    end
    Return root 
"""

critical_val = {1: 2.71, 2: 4.61, 3: 6.25, 4: 7.78, 5: 9.24}

# node or tree structure
# if node -> [ <attribute> , <split val> , <left> , <right>, <#examples_class> ] 
# if leaf -> [ -1 , label ]

def id3(data, labels, attributes, heuristic, prepruning):
    p = []
    for j in range(3):
        p.append(np.count_nonzero(labels==j))
    ## if all examples from same label -> return leaf with this label
    if np.all(labels == labels[0]):
        return [-1, labels[0],p]
    ## if there is no attribute more -> return leaf with most common label
    if np.shape(attributes)[0] == 0:
        counts = np.bincount(labels)
        return [-1, np.argmax(counts),p]
        
    ## select the proper attribute
    attr = -1
    max_gain = -1
    min_index = 99**9
    split = -1
    for a in attributes:
        x = dt.calculate_split_values(data,labels,3,a,heuristic)
        if heuristic == 'info_gain':
            i = x[np.argmax(x, axis=0)[1]]
            # i = [ splitvalx, x's gain]
            if i[1] > max_gain:
                attr = a
                max_gain = i[1]
                split = i[0]
        else:
            i = x[np.argmin(x, axis=0)[1]]
            if i[1] < min_index:
                attr = a
                min_index = i[1]
                split = i[0]

    # by splitting, create left and right buckets
    length = np.shape(data)[0]
    left_data = np.array([]).reshape(0,np.shape(data)[1])
    right_data = np.array([]).reshape(0,np.shape(data)[1])
    left_labels = np.array([]).reshape(0,1)
    right_labels = np.array([]).reshape(0,1)
    for i in range(length):
        if data[i][attr] < split:
            left_data = np.vstack( (left_data, data[i]) )
            left_labels = np.append( left_labels, labels[i] )
        else:
            right_data = np.vstack( (right_data, data[i]) )
            right_labels = np.append( right_labels, labels[i] )
    
    left_labels = left_labels.astype(int)
    right_labels = right_labels.astype(int)
    
    ## chi-square test for pruning
    l = []
    r = []   
    for j in range(3):
        l.append(np.count_nonzero(left_labels==j))
        r.append(np.count_nonzero(right_labels==j))
    chi, deg = dt.chi_squared_test(l,r)
    if chi < critical_val[deg] and prepruning:
        counts = np.bincount(labels)
        return [-1, np.argmax(counts),p]

    # if left or right node is empty -> return most common label
    # else recursively go left & right
    if np.shape(left_data)[0] == 0:
        counts = np.bincount(labels)
        left_node = [-1,np.argmax(counts),p]
    else:
        left_node = id3(left_data,left_labels,np.delete(attributes,np.where(attributes==attr)),heuristic,prepruning)
    if np.shape(right_data)[0] == 0:
        counts = np.bincount(labels)
        right_node = [-1,np.argmax(counts),p]
    else:
        right_node = id3(right_data,right_labels,np.delete(attributes,np.where(attributes==attr)),heuristic,prepruning)
    
    return [attr, split, p, left_node, right_node]

## it traverses tree to classify a sample
def classify(sample,tree):
    # if leaf -> return this label
    if tree[0] == -1:
        return tree[1]
    # if field lower than split value -> search left node
    if sample[tree[0]] < tree[1]:
        return classify(sample,tree[3])
    # if field bigger than split value -> search right node
    else:
        return classify(sample,tree[4])



train_set = np.load( 'dt/train_set.npy')
train_labels = np.load( 'dt/train_labels.npy')

tri1 = id3(train_set,train_labels,np.array([0,1,2,3]),'info_gain',False)
print(tri1)
tri2 = id3(train_set,train_labels,np.array([0,1,2,3]),'avg_gini_index',False)
print(tri2)
tri3 = id3(train_set,train_labels,np.array([0,1,2,3]),'info_gain',True)
print(tri3)
tri4 = id3(train_set,train_labels,np.array([0,1,2,3]),'avg_gini_index',True)
print(tri4)

trees = [tri1,tri2,tri3,tri4]

test_set = np.load('dt/test_set.npy')
test_labels = np.load('dt/test_labels.npy')

length = np.shape(test_set)[0]
# test all trees
for t in trees:
    accuracy = 0
    for i in range(length):
        if classify(test_set[i],t) == test_labels[i]:
            accuracy += 1
    print("Test accuracy: ", accuracy/length)
