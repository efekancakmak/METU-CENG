import nb

f = open('nb_data/train_set.txt','r')
train_set = f.read()
f.close()
train_set = train_set.split('\n')
length = len(train_set)

"""
- I saw cases like "I did it...@username" that decreases username features(valuable feature).
So I decided to keep punctuations away from words by adding ' ' between them.
- I think we generate new feature, number of spaces ' ', indicating total number of punctuations.
"""
some_punc = ['.', ',', '-', '?', '"', ';', '$']
for i in range(length):
    # for ex. "..." -> " .  .  . "
    for s in some_punc:
        # for ex. "@efekan..@user" -> "@efekan .  . @user"
        train_set[i] = train_set[i].replace(s,' ' + s + ' ')

splitted = [t.split(' ') for t in train_set]

vocabulary = nb.vocabulary(splitted)

# Read train labels
f = open('nb_data/train_labels.txt','r')
train_labels = f.read()
f.close()
train_labels = train_labels.split('\n')

# Calculate theta and pi
theta = nb.estimate_theta(splitted,train_labels,vocabulary)
del train_set
pi = nb.estimate_pi(train_labels)
del train_labels

# Read test set
f = open("nb_data/test_set.txt",'r')
test_lines = f.read()
f.close()
test_lines = test_lines.split('\n')

# Calculate scores on test sentences
test = nb.test(theta,pi,vocabulary,[t.split(' ') for t in test_lines])
del test_lines

# Read test labels
f = open("nb_data/test_labels.txt",'r')
test_labels = f.read()
f.close()
test_labels = test_labels.split('\n')
length = len(test_labels)

accuracy = 0
for i in range(length):
    res = max(test[i], key = lambda x: x[0])
    if test_labels[i] == res[1]:
        accuracy += 1
    
print("Accuracy on test set: ", accuracy/length)