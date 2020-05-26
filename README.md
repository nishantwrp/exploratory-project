## Exploratory Project

### Objective
The objective of this project is the early classification of a multivariate time series with different sampling rates while mantaining the `alpha` accuracy.

### Dataset

PEMS-SF Dataset is used in this project.
[Link](https://archive.ics.uci.edu/ml/datasets/PEMS-SF)

### Algorithms Used
- Gaussian Process Classifier
- Hidden Markov Model using viterbi algorithm

### Running the code

Execute the following commands to run this code
- `pipenv shell`
- `pipenv install`
- `python main.py`

### Summary
- I was able to achieve `95.45%` accuracy while using `85.48%` of the incoming data points of a time series.
- I used `80%` of the dataset for training and `20%` for testing.
- `ALPHA = 0.9`
- I used `10` sensors from the original dataset.

### File Structure
- **datasets/** - This contains the dataset used after pre processing. Each sensor has a different file.
- **mrls/** - This contains the mrls with different values of `alpha` and `training fraction`.
- **check_distribution.py** - This code was written to check if the labels are distributed evenly in the dataset.
- **gaussian_process_classifier.py** - This file contains all the code related to `gaussian_process_classifier`.
- **hidden_markov_model.py** - This file contains all the code related to `hidden_markov_model`.
- **load_dataset.py** - This file contains the functions to load the datasets in a format that they can be used from the `datasets` directory.
- **main.py** - This file contains the main code for this project.
- **pre_processing.py** - This file was used to convert the original dataset to the format present in the `datasets` directory.

### Final Output
```
SUMMARY
--------------------  ------------------------------------------------------------------------
Algorithms Used       Gaussian Process Classifier, Hidden Markov Model using viterbi algorithm
Accuracy              95.45%
Data Used             85.48%
Max data points used  134
Avg data points used  123.09
Sensors               10
Training fraction     0.8
Alpha                 0.9
--------------------  ------------------------------------------------------------------------

CONFUSION MATRIX
--  --  --  --  --  --  --
10   0   0   0   0   0   0
 0  10   0   0   0   0   0
 0   2  12   0   0   0   0
 0   0   0  12   0   0   0
 0   0   1   1  10   0   0
 0   0   0   0   0  13   0
 0   0   0   0   0   0  17
--  --  --  --  --  --  --

2 Wednesday(s) confused for Tuesday(s)
1 Friday(s) confused for Wednesday(s)
1 Friday(s) confused for Thursday(s)
```
