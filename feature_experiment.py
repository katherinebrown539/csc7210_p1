from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from read_data import read_data
import other_trees
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import decision_tree

def remove_features(removal_order, train_file, test_file, attr_file, max_features):
    train_accs = []
    test_accs = []
    remove_columns = []
    for col in removal_order:
        print(col)
        remove_columns.append(col)
        if len(remove_columns) == max_features: break
        print(remove_columns)
        train_data, train_attr = read_data(train, attr, remove_columns=remove_columns)
        test_data, test_attr = read_data(test, attr, remove_columns=remove_columns)
        tree = decision_tree.DecisionTreeLearning(train_data, train_attr, "normal", "class")

        decision_tree.print_tree(tree)
        y_pred, y_true = decision_tree.predict(train_data, tree)
        train_acc = decision_tree.accuracy_score(y_pred, y_true)
        print('Accuracy on Training Data: {0}'.format(train_acc*100))
        y_pred, y_true = decision_tree.predict(test_data, tree)
        test_acc = decision_tree.accuracy_score(y_pred, y_true)
        print('Accuracy on Training Data: {0}'.format( test_acc*100))

        train_accs.append(train_acc)
        test_accs.append(test_acc)
    return train_accs, test_accs

def remove_features_sklearn(criterion, splitter, removal_order, train_file, test_file, attr_file, max_features):
    train_accs = []
    test_accs = []
    remove_columns = []
    for col in removal_order:
        print(col)
        remove_columns.append(col)
        if len(remove_columns) == max_features: break
        print(remove_columns)
        train_data, train_attr = read_data(train, attr, remove_columns=remove_columns)
        test_data, test_attr = read_data(test, attr, remove_columns=remove_columns)
        train_acc, test_acc = other_trees.sklearn_decision_tree(criterion, splitter, train_data, test_data, train_attr, class_column, ds_name="custom")

    
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    return train_accs, test_accs


def visualize(train_accs, test_accs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(train_accs)))
    ax.plot(x, train_accs, label="Training Accuracy", marker='.')
    ax.plot(x, test_accs, label="Testing Accuracy", marker=".")
    ax.set_title("Decision Tree (SK-Entropy) Performance as Features are Removed")
    ax.set_xlabel("ith Feature Removed")
    ax.set_ylabel("Classifier Accuracy")
    ax.set_xticks(x)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    attr = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]
    
    train_data, train_attr = read_data(train, attr)
    class_column = "class"
    y = train_data[class_column]
    X = train_data.drop(class_column, axis=1, inplace=False)
    columns = list(X)
    print(X)
    enc = OrdinalEncoder()
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = enc.fit_transform(X)
    print(X)
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X,y)
    scores = fs.pvalues_.tolist()
    scores = pd.DataFrame([scores, columns]).T
    scores.columns = ["score", "feature"]
    scores = scores.sort_values(by="score")
    print(scores)
    removal_order = scores["feature"].values
    print(removal_order)

    # train_accs, test_accs = remove_features(removal_order, train, test, attr, len(columns))
    train_accs, test_accs = remove_features_sklearn(criterion="entropy", splitter="random", removal_order=removal_order, train_file=train, test_file=test, attr_file=attr, max_features=len(columns))
    visualize(train_accs, test_accs)