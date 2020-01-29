# Written by Nishant Mittal aka nishanwrp

# Imports
import csv
import os

# Classes
class Dataset:
    def __init__(self, name, features_path, labels_path):
        self.name = name
        self.features_path = features_path
        self.labels_path = labels_path

class feature_instance:
    def __init__(self, features):
        self.features = features

class features_data:
    def __init__(self, data_points):
        self.data_points = data_points

class label_instance:
    def __init__(self, label):
        self.label = label

# Constants

# There are 963 features present in the dataset, for simplicity we will only be dealing with 10
ALLOWED_FEATURES = 10

# Functions
def process_features_file(file_path):
    '''
    This function processes the features file and returns the array of all the instances object
    '''
    file_reader = open(file_path, 'r')
    instances = file_reader.readlines()

    instance_objects = list()

    for instance in instances:
        # Remove brackets []
        instance_string = instance[1: len(instance)-2]

        features = instance_string.split(';')
        feature_objets = list()

        feature_count = 0
        for feature in features:
            feature_object = features_data(feature.split(' '))
            feature_objets.append(feature_object)
            feature_count += 1

            if feature_count == ALLOWED_FEATURES:
                break

        instance_object = feature_instance(feature_objets)
        instance_objects.append(instance_object)

    return instance_objects

def process_labels_file(file_path):
    '''
    This function processes the labels file and returns the array of all the instances in a structured manner
    '''
    file_reader = open(file_path, 'r')
    file_content = file_reader.read()

    label_objects = list()

    # Remove brackets []
    labels_string = file_content[1: len(file_content)-2]
    labels = labels_string.split(' ')

    for label in labels:
        label_object = label_instance(label)
        label_objects.append(label_object)

    return label_objects

# Convert the dataset in csv format
dataset_paths = [Dataset('test_data', 'Dataset/PEMS_test', 'Dataset/PEMS_testlabels'), Dataset('train_data', 'Dataset/PEMS_train', 'Dataset/PEMS_trainlabels')]

complete_features_list = list()
complete_labels_list = list()

for dataset in dataset_paths:
    features_list = process_features_file(dataset.features_path)
    labels_list = process_labels_file(dataset.labels_path)

    complete_features_list += features_list
    complete_labels_list += labels_list

# Write datasets for each feature
current_path = os.getcwd()
directory_name = "datasets"

# Create a directory named <directory_name>
os.makedirs(os.path.join(current_path, directory_name))

for feature_number in range(ALLOWED_FEATURES):
    csv_file_name = str(feature_number+1) + ".csv"
    csv_file_path = directory_name + "/" + csv_file_name

    with open(csv_file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        for i, feature_object in enumerate(complete_features_list):
            row = list()

            feature = feature_object.features[feature_number]
            data_points = feature.data_points

            row += data_points
            row.append(complete_labels_list[i].label)

            csv_writer.writerow(row)
