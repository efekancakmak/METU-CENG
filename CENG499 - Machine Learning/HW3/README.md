First, I implemented dt.py

In id3.py, I implemented id3 algorithm by using functions in dt.py
    If you run this python file:
        It will print trees on STDOUT
            info gain w/out chi-square
            avg gini index w/out chi-square
            info gain with chi-square
            avg gini index with chi-square
            respectively.
        Then,
        It will print test accuracies of them in the same order.

TO visualize trees, I implemented treelib.py
This executable trains sklearn tree and saves .dot file of tree namely out.dot
Then I used this file as template -> converted into mydot.dot and mydot2.dot by hand 
(looking STDOUT of my trees)

mydot.dot is the file of the second tree in my report
mydot2.dot is the file of the first tree in my report

I used these two commands below to create png's from .dot files
dot -Tpng mydot2.dot -o tree.png
dot -Tpng mydot.dot -o tree.png



SVM
If you execute svm.py
    first, it will show Task1 plots, then Task2 plots
    secondly, it will print(STDOUT) hyperparameter configurations for Task3
    lastly, it will print(STDOUT) confusion matrix fields for Task4
