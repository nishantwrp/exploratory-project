# Written by Nishant Mittal aka nishantwrp

# Imports
from load_dataset import load_single_dataset
from tabulate import tabulate

# Constants
ROWS_FRACTION = 0.8
LABELS = 7

# Code
mts, labels = load_single_dataset(1)

rows_to_be_processed = int(ROWS_FRACTION*len(labels))

labels = labels[:rows_to_be_processed]

labels_count = [ 0 for _ in range(LABELS) ]
for label in labels:
    labels_count[label-1] += 1

table = list()
for i in range(LABELS):
    table.append([i+1, labels_count[i]])

table.append(['Total', rows_to_be_processed])

print(tabulate(table, headers=['Label', 'Count'], tablefmt="grid"))
