# CSC 7210: Project 1
## Katherine Brown

### Introduction
Please run my code on a Unix system. Although the code should work on Windows, it has been verified to run on MacOS and Linux.

### About
This is a custom implementation of the Iterative Dichotomiser 3 (ID3) algorithm for generating decision trees. This project used a custom version of the KDD Cup 1999 dataset to detect various cyber attacks on a network system. 

### Anaconda
To run my code, you will need to install several Python libraries. I recommend using Anaconda's virtual environments. Otherwise, feel free to skip to the "Package Installation" section

Anaconda can be installed from https://www.anaconda.com/products/individual

This PDF contains additional information on using Conda Virtual Environments: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf


After successfully installing Anaconda, run the following in the command line to create a virtual environment with the following command:
`conda create --name csc7210 python=3.7.4`

You will then need to activate the environment to further install packages and run the code. To do so, run
`conda activate csc7210`

### Package Installation
You will need to install the following packages to run any file in the submission. Scikit-Learn and Matplotlib are used to implement the decision trees that my implementation is to be compared to.

`conda install pandas=0.25.1`
`conda install numpy=1.17.2`
`conda install matplotlib`
`conda install scikit-learn`

### Running My Code
I have three executable files

`decision_tree.py` contains my decision tree implementation. It contains a driver that prints out the required information. To run the code execute the following command

`python decision_tree.py attr_file training_data testing_data`

You will need to replace "attr_file" with the path to the attributes file, "training data" with the path to the training data, and "testing data" with the path to the testing data.

`other_trees.py` contains the SciKit-Learn tests. It contains a driver that prints out the required information. To run the code execute the following command

`python other_trees.py attr_file training_data testing_data`

You will need to replace "attr_file" with the path to the attributes file, "training data" with the path to the training data, and "testing data" with the path to the testing data.

Finally, `feature_experiment.py` contains a brief experiment where I tried to rank features using the ANOVA F-Test and remove from most important to least. Unfortunately, the results were honestly uninteresting and I opted for a different route for my report. To run the code execute the following command

`python feature_experiment.py attr_file training_data testing_data`

You will need to replace "attr_file" with the path to the attributes file, "training data" with the path to the training data, and "testing data" with the path to the testing data.

This should be all the information needed to run my code successfully. If something doesn't work right, please email me at kebrown46@tntech.edu. Thanks!
