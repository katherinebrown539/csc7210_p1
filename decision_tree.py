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
    
    
    print(examples[class_column].unique().shape[0])


if __name__ == "__main__":
    train_data, train_attr = read_data("data/ids-train.txt", "data/ids-attr.txt")
    DecisionTreeLearning(train_data, train_attr, 0, "class")