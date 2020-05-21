# Written by Nishant Mittal aka nishantwrp

# Imports
from load_dataset import load_all_datasets
from progress.bar import IncrementalBar
from gaussian_process_classifier import calculate_mrl, get_gpc
from time import time
import pickle
import json
import os

# Constants
ALPHA = 0.9
INITIAL_F = 0.4
LABELS = 7
TRAINING_FRACTION = 0.8

# Main Code Begins Here
all_datasets = load_all_datasets()
all_mrls = list()

if os.path.exists('mrls.json'):
    with open('mrls.json', 'r') as file:
        all_mrls_dict = json.load(file)

        # Convert all_mrls to array
        for key, mrl_data_dict in all_mrls_dict.items():
            mrl_array = list()
            for key, mrl_val in mrl_data_dict.items():
                mrl_array.append(mrl_val)
            all_mrls.append(mrl_array)

        print("Precalculated MRL data loaded")
else:
    for i, dataset in enumerate(all_datasets):
        print("MRL Calculation for feature %s started" % (str(i+1)))

        t = time()

        X, Y = dataset

        # Training dataset
        rows_for_training = int(TRAINING_FRACTION*X.shape[0])
        X = X[:rows_for_training]
        Y = Y[:rows_for_training]

        initial_f = int(INITIAL_F*X.shape[1])
        mrl = calculate_mrl(X, Y, ALPHA, initial_f)

        time_elapsed = str(round(time() - t, 3)) + "s"
        print("MRL for feature %s calculated\nTime elapsed: %s\n" % (str(i+1), time_elapsed))

        all_mrls.append(mrl)

    # Convert all_mrls To Dict
    all_mrls_dict = dict()

    for i, mrl in enumerate(all_mrls):
        # Further convert mrl to dict
        mrl_dict = dict()
        for j, mrl_data in enumerate(mrl):
            mrl_dict[j+1] = mrl_data

        all_mrls_dict[i+1] = mrl_dict

    # Save the mrls as json
    with open('mrls.json', 'w+') as file:
        json.dump(all_mrls_dict, file)
        print("Saved the calculated MRLs for future use")


trained_classifiers = list()

if os.path.exists('trained_models'):
    with IncrementalBar('Loading gaussian process classifiers', max=len(all_datasets)) as bar:
        t = time()

        for i in range(len(all_datasets)):
            with open('trained_models/model_' + str(i+1), 'rb') as f:
                trained_classifiers.append(pickle.load(f))
            bar.next()

        time_elapsed = str(round(time() - t, 3)) + "s"
        print("\nTime elapsed: %s" % (time_elapsed))

else:
    os.makedirs('trained_models')
    with IncrementalBar('Training guassian process classifiers', max=len(all_datasets)) as bar:
        t = time()

        for i, dataset in enumerate(all_datasets):
            # Training dataset
            X, Y = dataset
            rows_for_training = int(TRAINING_FRACTION*X.shape[0])
            X = X[:rows_for_training]
            Y = Y[:rows_for_training]

            mrl = int(min([min(all_mrls[i]) for i in range(len(all_datasets))]))
            X = X[:, :mrl]

            trained_classifiers.append(get_gpc(X, Y))
            with open('trained_models/model_' + str(i+1), 'wb') as f:
                pickle.dump(trained_classifiers[-1], f)
            bar.next()

        time_elapsed = str(round(time() - t, 3)) + "s"
        print("\nTime elapsed: %s" % (time_elapsed))


print("Calculating transition matrix")
predicted_probs = list()

for i, dataset in enumerate(all_datasets):
    X, Y = dataset
    rows_for_training = int(TRAINING_FRACTION*X.shape[0])
    X = X[:rows_for_training]
    Y = Y[:rows_for_training]

    mrl = int(min([min(all_mrls[i]) for i in range(len(all_datasets))]))
    X = X[:, :mrl]

    predicted_probs.append(trained_classifiers[i].predict(X))

transition_matrix = [[0 for _ in range(LABELS)] for _ in range(LABELS)]

for i in range(1, len(predicted_probs)):
    for j in range(len(predicted_probs[i])):
        transition_matrix[predicted_probs[i-1][j] - 1][predicted_probs[i][j] - 1] += 1

for i, row in enumerate(transition_matrix):
    total = sum(row)
    for j in range(len(row)):
        transition_matrix[i][j] /= total

print("Calculated transition matrix")