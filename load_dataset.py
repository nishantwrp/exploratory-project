# Written by Nishant Mittal aka nishantwrp

# Imports
import pandas as pd
import os

# Constants
ALLOWED_FEATURES = 10
DATASETS_DIRECTORY = 'datasets'

def load_single_dataset(feature):
    '''
    Loads a single dataset corresponding to the given feature in format mts, labels
    '''
    file_name = str(feature) + '.csv'
    file_path = os.path.join(DATASETS_DIRECTORY, file_name)

    dataset = pd.read_csv(file_path, header=None)
    no_of_columns = dataset.shape[1]

    labels = dataset[no_of_columns - 1].to_numpy()
    mts = dataset.iloc[:, :-1].to_numpy()

    return mts, labels

def load_all_datasets():
    '''
    Loads all datasets
    '''
    datasets = list()

    for i in range(1, ALLOWED_FEATURES+1):
        datasets.append(load_single_dataset(i))

    return datasets