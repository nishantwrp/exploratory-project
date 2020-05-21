# Written by Nishant Mittal aka nishantwrp

# Imports
from load_dataset import load_all_datasets
from progress.bar import IncrementalBar
from gaussian_process_classifier import calculate_mrl, get_gpc
from hidden_markov_model import prepare_observation_matrix, HMM
from time import time
from tabulate import tabulate
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

        print("Precalculated MRL data loaded\n")
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

# Delete the models if mrls are changed
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


transition_matrix = [[0 for _ in range(LABELS)] for _ in range(LABELS)]

with IncrementalBar('Calculating transition matrix', max=1) as bar:
    t = time()
    predicted_probs = list()

    for i, dataset in enumerate(all_datasets):
        X, Y = dataset
        rows_for_training = int(TRAINING_FRACTION*X.shape[0])
        X = X[:rows_for_training]
        Y = Y[:rows_for_training]

        mrl = int(min([min(all_mrls[i]) for i in range(len(all_datasets))]))
        X = X[:, :mrl]

        predicted_probs.append(trained_classifiers[i].predict(X))

    for i in range(1, len(predicted_probs)):
        for j in range(len(predicted_probs[i])):
            transition_matrix[predicted_probs[i-1][j] - 1][predicted_probs[i][j] - 1] += 1

    for i, row in enumerate(transition_matrix):
        total = sum(row)
        for j in range(len(row)):
            transition_matrix[i][j] /= total

    bar.next()
    time_elapsed = str(round(time() - t, 3)) + "s"
    print("\nTime elapsed: %s" % (time_elapsed))


TRAINING_ROWS = int(len(all_datasets[0][0])*TRAINING_FRACTION)
TESTING_ROWS = len(all_datasets[0][0]) - TRAINING_ROWS
testing_data = [[None for _ in range(len(all_datasets))] for _ in range(TESTING_ROWS)]
testing_labels = list()

with IncrementalBar('Preparing the test dataset', max=len(all_datasets)) as bar:
    t = time()

    for i, dataset in enumerate(all_datasets):
        X, Y = dataset
        X = X[TRAINING_ROWS:]
        Y = Y[TRAINING_ROWS:]

        mrl = int(min([min(all_mrls[i]) for i in range(len(all_datasets))]))
        X = X[:, :mrl]

        for j, row in enumerate(X):
            testing_data[j][i] = row

        testing_labels = Y
        bar.next()

    time_elapsed = str(round(time() - t, 3)) + "s"
    print("\nTime elapsed: %s" % (time_elapsed))


predicted_labels = list()

with IncrementalBar('Predicting the labels using viterbi algorithm', max=TESTING_ROWS) as bar:
    t = time()

    for testing_row in testing_data:
        predicted_probs = list()
        for i, testing_component in enumerate(testing_row):
            predicted_probs.append(trained_classifiers[i].predict_proba([testing_component])[0])
        observation_matrix = prepare_observation_matrix(predicted_probs)

        hidden_markov_model = HMM(transition_matrix, observation_matrix)
        predicted_seq = hidden_markov_model.predict_state_sequence(len(all_datasets))
        predicted_label = (max(set(predicted_seq), key=predicted_seq.count) + 1)
        predicted_labels.append(predicted_label)
        bar.next()

    time_elapsed = str(round(time() - t, 3)) + "s"
    print("\nTime elapsed: %s" % (time_elapsed))


accuracy = 0

for i in range(len(predicted_labels)):
    if predicted_labels[i] == testing_labels[i]:
        accuracy += 1/len(predicted_labels)

mrl = int(min([min(all_mrls[i]) for i in range(len(all_datasets))]))

print("SUMMARY")
summary = [
    ["Algorithms Used", "Gaussian Process Classifier, Hidden Markov Model using viterbi algorithm"],
    ["Accuracy", "{}%".format(str(round(accuracy*100, 2)))],
    ["Data Used", "{}%".format(str(round((mrl/144)*100, 2)))],
    ["Sensors", len(all_datasets)],
    ["Training fraction", TRAINING_FRACTION],
    ["Alpha", ALPHA]
]
print(tabulate(summary))
