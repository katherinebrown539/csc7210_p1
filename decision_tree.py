import pandas as pd
import numpy as np
from read_data import read_data
from collections import Counter, OrderedDict
import pprint

def DecisionTreeLearning(examples, attributes, default, class_column):
    """
        Implementation of Decision Tree Learning as given by Russell/Norvig Artificial Intelligence: A modern approach

        Parameters:
            * examples: a pandas dataframe of usable examples
            * attributes: a python dataframe of attributes and their list of possible values
            * default: default value of tree
            * class_column: the column in the dataframe that contains the class
    """
    print("# examples: ", examples.shape[0])
    print("# classes: ", examples[class_column].unique().shape[0])
    print("available attributes: ", len(attributes))
    
    if examples.shape[0] == 0: 
        print("no examples left")
        return default
    elif examples[class_column].unique().shape[0] == 1: #have the same classification
        print("all examples are same class")
        return examples[class_column].unique()[0] 
    elif len(attributes.keys()) == 0:
        print("no attributes left")
        return MajorityValue(examples, class_column) 
    else:
        best = ChooseAttribute(attributes, examples, class_column)
        print("Chosen Attribute: ", best)
        tree = {}
        tree["attribute_name"]= best
        values = attributes[best]
        del attributes[best]
        for vi in values:
            examples_i = examples[examples[best] == vi]
            # print(examples_i)
            tree[vi] = DecisionTreeLearning(examples_i, attributes, default, class_column)
    return tree
    
def MajorityValue(examples, class_column):
    classes = examples[class_column]
    return Counter(classes).most_common(1)

def ChooseAttribute(attributes, examples, class_column):
    max_info_gain = 0
    attribute = None
    for key in attributes.keys():
        if key == class_column: continue
        class_entropy = calc_entropy(values=examples[class_column])
        class_entropy_attr = calc_entropy_attr(examples, key, class_column)
        ig = class_entropy - class_entropy_attr
        if ig > max_info_gain:
            max_info_gain = ig
            attribute = key
    return attribute

def calc_entropy(values):
    values, counts = np.unique(values, return_counts=True)
    entropy = 0
    total = np.sum(counts)
    for count in counts:
        entropy += (-1)* (count/total) * np.log2(count/total)
    return entropy

def calc_entropy_attr(examples, attribute, class_column):
    values, counts = np.unique(examples[attribute], return_counts=True)
    total = np.sum(counts)
    entropy = 0
    index = 0
    for value in counts:
        weight = (value/total)
        examples_attr = examples[examples[attribute] == values[index]]
        attribute_value_entropy = calc_entropy(examples_attr[class_column])
        entropy += weight * attribute_value_entropy
        index += 1
    return entropy

def print_tree(tree):
    # pp = pprint.PrettyPrinter(indent=4)
    pprint.pprint(tree, indent=4)

def predict_point(y, tree):
    if isinstance(tree, str): return tree
    attr = tree['attribute_name']
    y_attr = y[attr]
    return predict_point(y, tree[y_attr])

def predict(test_data, tree, class_column="class"):
    predictions = []
    for index, row in test_data.iterrows():
        pred = predict_point(row, tree)
        predictions.append(pred)
    test_data["prediction"] = predictions
    test_data.to_csv("test.csv")
    return test_data["prediction"].values, test_data[class_column].values

def accuracy_score(y_pred, y_true):
    if y_pred.shape[0] != y_true.shape[0]: return -1
    total = 0
    correct = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i]: correct += 1
        total += 1
    return correct/total

if __name__ == "__main__":
    train_data, train_attr = read_data("data/ids-train.txt", "data/ids-attr.txt")
    test_data, test_attr = read_data("data/ids-test.txt", "data/ids-attr.txt")
    # train_data, train_attr = read_data("data/restaurant_train.txt", "data/restaurant_attr.txt")
    # test_data, test_attr = read_data("data/restaurant_test.txt", "data/restaurant_attr.txt")
    tree = DecisionTreeLearning(train_data, train_attr, 0, "class")
    print_tree(tree)
    y_pred, y_true = predict(test_data, tree)
    print(accuracy_score(y_pred, y_true))
