import pandas as pd

def read_data(data_file, attributes_file, delim=" "):
    '''
        This function opens a data file and stores it in a pandas dataframe that is returned.
        The file provided must be space-delimited

        Parameters:
            * data_file: the path to the data file; either give the file path relative to the working directory or the full path name
            * attributes_file: the path to the file containing the names of the attributes
            * delim: default " "; how attributes of a record in the file are delimited 
    '''
    attrs = []
    attr_dict = {}
    with open(attributes_file) as f:

        for line in f:
            line = line.split(" ")
            print(line)
            attr_name = line[0]
            attrs.append(attr_name)
            attr_values = line[1:]
            attr_values = [val.strip('\n') for val in attr_values]
            attr_dict[attr_name] = attr_values 


    data = pd.read_csv(data_file, delimiter=delim, header=None)
    data.columns = attrs
    return data, attr_dict



if __name__ == "__main__":
    df, attr_dict = read_data("data/ids-train.txt", "data/ids-attr.txt")
    print(df)
    print(attr_dict)
    df, attr_dict = read_data("data/ids-test.txt", "data/ids-attr.txt")
    print(df)