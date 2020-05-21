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
- I was able to achieve `89.7%` accuracy while using only `39.5%` of the incoming data points of a time series.
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
