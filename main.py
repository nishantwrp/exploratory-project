# Written by Nishant Mittal aka nishantwrp

# Imports
from load_dataset import load_all_datasets
from gaussian_process_classifier import calculate_mrl
from time import time
import json
import os

# Constants
ALPHA = 0.8
INITIAL_F = 0.4
TRAINING_FRACTION = 0.7

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
