# Written by Nishant Mittal aka nishantwrp

# Imports
from load_dataset import *
from gaussian_process_classifier import calculate_mrl
from time import time

# Constants
ALPHA = 0.8
INITIAL_F = 0.7

# Main Code Begins Here
all_datasets = load_all_datasets()
all_mrls = list()

for i in range(len(all_datasets)):
    print("MRL Calculation for feature %s started" % (str(i+1)))

    t = time()

    X, Y = all_datasets[i]
    
    initial_f = int(INITIAL_F*X.shape[1])
    if len(all_mrls) > 0:
        initial_f = max(all_mrls)

    mrl = calculate_mrl(X, Y, ALPHA, initial_f)

    time_elapsed = str(round(time() - t, 3)) + "s"
    print("MRL for feature %s: %s\nTime elapsed: %s\n" % (str(i+1), str(mrl), time_elapsed))

    all_mrls.append(mrl)

max_mrl = max(all_mrls)
print("The final MRL value is %s" % str(max_mrl))