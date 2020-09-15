import pandas as pd
import numpy as np
from read_data import read_data

def DecisionTreeLearning(examples, attributes, default, class_column):
    """
        Implementation of Decision Tree Learning as given by Russell/Norvig Artificial Intelligence: A modern approach

        Parameters:
            * examples: a pandas dataframe of usable examples
            * attributes: a python dataframe of attributes and their list of possible values
            * default: default value of tree
            * class_column: the column in the dataframe that contains the class
    """
    print("# attributes: ", examples.shape[0])
    print("# classes: ", examples[class_column].unique().shape[0])
    print("available attributes: ", len(attributes))
    print()
    if examples.shape[0] == 0: return default
    elif examples[class_column].unique().shape[0] == 1: #have the same classification
        return examples[class_column].unique()[0] 
    elif len(attributes.keys()) == 0:
        return MajorityValue(examples, class_column) 
    else:
        best = ChooseAttribute(attributes, examples, class_column)
        tree = {"attribute_name": best}
        values = attributes[best]
        del attributes[best]
        for vi in values:
            examples_i = examples[examples[best] == vi]
            tree[vi] = DecisionTreeLearning(examples_i, attributes, default, class_column)
            print(tree)
    return tree
    
def MajorityValue(examples, class_column):
    classes = examples[class_column]
    return Counter(classes).most_common(1)

def ChooseAttribute(attributes, examples, class_column):
    max_info_gain = 0
    attribute = None
    for key in attributes.keys():
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
if __name__ == "__main__":
    train_data, train_attr = read_data("data/ids-train.txt", "data/ids-attr.txt")
    tree = DecisionTreeLearning(train_data, train_attr, 0, "class")
    print(tree)