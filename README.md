# 42-logistic-regression

The aim of this project is to implement from scratch a logistic regression algorithm to predict the Hogwarts house of students based on their course performance.
This is a multi-class classification problem.

### Installation

Make sure your Python version is **3.7**

### Components

#### Visualisation
- `src/describe.py`:
- `src/histogram.py`:
- `src/pair_plot.py`:
- `src/scatter_plot.py` : this script finds the two courses that have the closest distribution in each of the four Hogwarts houses and plots the corresponding distributions.

To run the script on the Hogwarts training dataset:

```
.src/scatter_plot.py data/dataset_train.csv
```

#### Classification
- `src/logistic_regression.py`: this file contains the logistic regression class definition. The API is based on Scikit-learn base classifier API.

To train the model:

```
$ src/train.py --help
usage: train.py [-h] -i INPUT -o OUTPUT [-p]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The file containing the training data
  -o OUTPUT, --output OUTPUT
                        Output dir to store the classifier and the encoders
  -p, --plot            Print graphs after training
$ src/train.py -i data/dataset_train.csv -o output -p
...
```

To make predictions:

```
$ src/predict.py --help
usage: predict.py [-h] -i INPUT -m MODEL -e ENCODER -s SCALER -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the test data
  -m MODEL, --model MODEL
                        Path to the trained model
  -e ENCODER, --encoder ENCODER
                        Path to the trained encoder
  -s SCALER, --scaler SCALER
                        Path to the trained scaler
  -o OUTPUT, --output OUTPUT
                        Path to save the predictions
$ src/predict.py -i data/dataset_test.csv -m output/model -e output/encoder -s output/scaler -o output/houses.csv
...
```

### Authors

- Sami Mhirech
- Jean-Eudes Peloye
- Guilhem Vuillier
