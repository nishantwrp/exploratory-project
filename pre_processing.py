# Written by Nishant Mittal aka nishanwrp

# Imports
import csv

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

        # There are 963 features present in the dataset, for simplicity we will only be dealing with 10
        ALLOWED_FEATURES = 10

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

for dataset in dataset_paths:
    features_list = process_features_file(dataset.features_path)
    labels_list = process_labels_file(dataset.labels_path)

    # Write dataset
    csv_file_name = dataset.name + ".csv"
    with open(csv_file_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        for i in range(len(features_list)):
            row = list()
            for feature in features_list[i].features:
                x = ","
                data_points_string = x.join(feature.data_points)
                row.append(data_points_string)
            row.append(labels_list[i].label)

            csv_writer.writerow(row)

# Combine both testing and training dataset
test_file_path = dataset_paths[0].name + ".csv"
train_file_path = dataset_paths[1].name + ".csv"
dataset_path = "dataset.csv"

test_file = open(test_file_path, "r")
test_file_data = test_file.read()

train_file = open(train_file_path, "r")
train_file_data = train_file.read()

dataset_file = open(dataset_path, "w+")
dataset_file.write(test_file_data)
dataset_file.write(train_file_data)

