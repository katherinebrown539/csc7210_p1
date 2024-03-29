import pandas as pd
import numpy as np
from read_data import read_data
from collections import Counter, OrderedDict
import pprint
import json
import sys

def DecisionTreeLearning(examples, attributes, default, class_column):
    """
        Implementation of Decision Tree Learning as given by Russell/Norvig Artificial Intelligence: A modern approach

        Parameters:
            * examples: a pandas dataframe of usable examples
            * attributes: a python dataframe of attributes and their list of possible values
            * default: default value of tree
            * class_column: the column in the dataframe that contains the class
        
        Returns:
            * A decision tree implemented as a Python dictionary 
    """
    # print("# examples: ", examples.shape[0])
    # print("# classes: ", examples[class_column].unique().shape[0])
    # print("available attributes: ", len(attributes))
    
    if examples.shape[0] == 0: 
        # print("no examples left")
        return default
    elif examples[class_column].unique().shape[0] == 1: #have the same classification
        # print("all examples are same class")
        return examples[class_column].unique()[0] 
    elif len(attributes.keys()) == 1:
        # print("no attributes left")
        return MajorityValue(examples, class_column) 
    else:
        best = ChooseAttribute(attributes, examples, class_column)
        print("Chosen Attribute: ", best)
        tree = OrderedDict()
        tree["attribute_name"]= best
        values = attributes[best]
        del attributes[best]
        for vi in values:
            examples_i = examples[examples[best] == vi]
            # print(examples_i)
            tree[vi] = DecisionTreeLearning(examples_i, attributes, default, class_column)
    return tree
    
def MajorityValue(examples, class_column):
    """
        Helper function to determine the mode of the class_column in the dataframe examples

        Parameters:
            * examples: a pandas dataframe of usable examples
            * class_column: the column that contains the class values of data instances
        
        Returns:
            * The class that appears the most in the examples dataframe
    """
    classes = examples[class_column]
    majority = Counter(classes).most_common(1)
    print(type(majority))
    return majority[0][0]

def ChooseAttribute(attributes, examples, class_column):
    """
        Helper function that chooses an attribute of the of the data set that maximizes information gain 

        Parameters: 
            * attributes: a Python dictionary of attributes that contains attribute names as keys and a list of possible attribute values as data
            * examples: a pandas dataframe of usable examples
            * class_column: the column that contains the class values of data instances
        
        Returns:
            * attribute name that maximizes information gain to be a split attribute
    """
    max_info_gain = -100
    attribute = None
    print(attributes.keys())
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
    """
        Calculates entropy of an array of values

        Parameters: 
            * values: a numpy array of values of which to calculate entropy
        
        Returns:
            * entropy: the entropy of the array
    """
    values, counts = np.unique(values, return_counts=True)
    entropy = 0
    total = np.sum(counts)
    for count in counts:
        entropy += (-1)* (count/total) * np.log2(count/total)
    return entropy

def calc_entropy_attr(examples, attribute, class_column):
    """
        Calculates entropy of the attribute provided
        Calculation is based on Russell and Norvig, 3rd edition, pp. 703 - 704

        Parameters: 
            * attributes: a Python dictionary of attributes that contains attribute names as keys and a list of possible attribute values as data
            * examples: a pandas dataframe of usable examples
            * class_column: the column that contains the class values of data instances
        
        Returns: 
            * Entropy of attribute
    """
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

def print_tree(tree, depth=0):
    """
        Prints the provided decision tree. 
        This method is recursive. 

        Parameters:
            * tree: tree/subtree to print
            * depth: current depth; indicates how much to indent tree printout
    """
    print("\t"*depth, "attribute_name: ", tree["attribute_name"])
    for key, value in tree.items():
        if key == "attribute_name": continue
        if isinstance(value, str):
            print("\t"*depth, key, ": ", tree[key])
        elif isinstance(value, OrderedDict):
            print("\t"*depth, key, ":")
            print_tree(value, depth+1)
    
def predict_point(y, tree):
    """
        Predicts the classification for a single data instance
        This method is recursive until a prediction is determined.

        Parameters:
            * y: data point to determine class
            * tree: subtree of which to determine the next route in the tree; 
                    when this is a string, the recursion terminates as this is the class
    """
    if isinstance(tree, str): return tree
    attr = tree['attribute_name']
    y_attr = y[attr]
    return predict_point(y, tree[y_attr])

def predict(test_data, tree, class_column="class"):
    """
        Generates predictions for a dataframe of data instances

        Parameters:
            * test_data: the dataframe of test instances and their true classes
            * tree: decision tree to use for predictions
            * class_column: column of correct classes 

        Returns:
            * numpy array of predictions
            * numpy array of correct values
    """
    tree = json.loads(json.dumps(tree))
    predictions = []
    for index, row in test_data.iterrows():
        pred = predict_point(row, tree)
        predictions.append(pred)
    test_data["prediction"] = predictions
    test_data.to_csv("test.csv")
    return test_data["prediction"].values, test_data[class_column].values

def accuracy_score(y_pred, y_true):
    """
        calculates accuracy of predictions and true values

        Parameters:
            * y_pred: a numpy array of predictions 
            * y_true: a numpy array of predictions
        
        Returns:
            * real value between 0 and 1 for the accuracy of the predictions to the true value
    """
    if y_pred.shape[0] != y_true.shape[0]: return -1
    total = 0
    correct = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i]: correct += 1
        total += 1
    return correct/total

if __name__ == "__main__":
    attr = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]
    
    remove_columns = None 
    train_data, train_attr = read_data(train, attr, remove_columns=remove_columns)
    test_data, test_attr = read_data(test, attr, remove_columns=remove_columns)
    tree = DecisionTreeLearning(train_data, train_attr, "normal", "class")

    print_tree(tree)
    y_pred, y_true = predict(train_data, tree)
    train_acc = accuracy_score(y_pred, y_true)
    print('Accuracy on Training Data: {0}'.format(train_acc*100))
    y_pred, y_true = predict(test_data, tree)
    test_acc = accuracy_score(y_pred, y_true)
    print('Accuracy on Testing Data: {0}'.format( test_acc*100))