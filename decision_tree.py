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

    if examples.shape[0] == 0: return default
    elif examples[class_column].unique().shape[0] == 1: #have the same classification
        return examples[class_column].unique()[0] #by return, that may mean return tree node
    elif len(attributes.keys()) == 0:
        return MajorityValue(examples) #by return, that may mean return tree node
    else:
        best = ChooseAttribute(attributes, examples)
        tree = None #create new decision tree 
    
def MajorityValue(examples, class_column):
    classes = examples[class_column]
    return Counter(classes).most_common(1)

def ChooseAttribute(attributes, examples):
    return None

if __name__ == "__main__":
    train_data, train_attr = read_data("data/ids-train.txt", "data/ids-attr.txt")
    DecisionTreeLearning(train_data, train_attr, 0, "class")