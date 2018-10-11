# 42-logistic-regression

The aim of this project is to implement from scratch a logistic regression algorithm to predict the Hogwarts house of students based on their course performance.
This is a multi-class classification problem.

### Installation

Make sure your Python version is **3.7**

### Components

#### Visualisation

- `src/describe.py`: this script gives a complete description of each of your numerical features (mean, std, max, min, median, first and third quartiles).

To run the script on the training Hogwarts dataset:

```
./src/describe.py data/dataset_train.csv
```


- `src/histogram.py`: this script finds the courses that have a similar distribution of grades for each of the Hogwarts houses. In order to do so, for each course, the script executes  Kolmogorov-Smirnov to test the similarity of distribution between two houses for all combination of houses. If all the p-values of these tests are above 5%, the course is said to have a homogeneous distribution and the program plots histograms of its distribution for each house.

To run the script on the training Hogwarts dataset:

```
./src/histogram.py data/dataset_train.csv
```


- `src/pair_plot.py`: this script displays the scatter-plot matrix for the grades of all courses at Hogwarts.

To run the script on the Hogwarts training dataset:

```
.src/pair_plot.py data/dataset_train.csv
```


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
