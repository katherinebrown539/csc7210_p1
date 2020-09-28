import sklearn.tree as trees
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from read_data import read_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def accuracy_score(y_pred, y_true):
    if y_pred.shape[0] != y_true.shape[0]: return "ERROR"
    total = 0
    correct = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i]: correct += 1
        total += 1
    print("{0}/{1}".format(correct, total))
    return correct/total

def visualize_tree(tree, fn, cn, filename):
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    trees.plot_tree(tree,
                    feature_names = fn, 
                    class_names=cn,
                    filled = True)
    fig.savefig(filename)

def sklearn_decision_tree(criterion, splitter, train_data, test_data, attr, class_column, ds_name):
    enc = OrdinalEncoder()
    le = LabelEncoder()
    y_train = train_data[class_column]
    y_train = le.fit_transform(y_train)
    X_train = train_data.drop([class_column], axis=1)
    X_train = enc.fit_transform(X_train)
    y_test = test_data[class_column]
    y_test = le.transform(y_test)
    X_test = test_data.drop([class_column], axis=1)
    X_test = enc.transform(X_test)
    tree = trees.DecisionTreeClassifier(criterion=criterion, splitter=splitter)

    #fit on training data
    tree.fit(X_train, y_train)
    features = list(attr.keys())
    features.remove(class_column)
    #plot tree
    visualize_tree(tree, fn=features, cn=attr["class"], filename="{0}_{1}_{2}.png".format(ds_name, criterion,splitter))


    #get training accuracy
    y_pred = tree.predict(X_train)
    train_acc = accuracy_score(y_pred, y_train)
    print('Accuracy on Training Data ({0}): {1}'.format(ds_name, train_acc*100))
    #get testing accuracy
    y_pred_test = tree.predict(X_test)
    test_acc = accuracy_score(y_pred_test, y_test)
    print('Accuracy on Testing Data ({0}): {1}'.format(ds_name, test_acc*100))
    return train_acc, test_acc

if __name__ == "__main__":

    attr = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

    remove_columns = ["srv_serror_rate", "serror_rate", "count", "same_srv_rate", "srv_diff_host_rate", "srv_count"]
    train_data, train_attr = read_data(train, attr, remove_columns=remove_columns)
    test_data, test_attr = read_data(test, attr, remove_columns=remove_columns)
    sklearn_decision_tree(criterion="entropy", splitter="best", train_data=train_data, test_data=test_data, attr=train_attr, class_column="class", ds_name="custom")
    sklearn_decision_tree(criterion="gini", splitter="best", train_data=train_data, test_data=test_data, attr=train_attr, class_column="class", ds_name="custom")
    sklearn_decision_tree(criterion="entropy", splitter="random", train_data=train_data, test_data=test_data, attr=train_attr, class_column="class", ds_name="custom")
    sklearn_decision_tree(criterion="gini", splitter="random", train_data=train_data, test_data=test_data, attr=train_attr, class_column="class", ds_name="custom")
