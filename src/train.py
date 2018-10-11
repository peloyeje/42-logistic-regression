#! /usr/bin/env python

import argparse
import pickle
import sys

from logistic_regression import LogisticRegression
from utils import OneHotEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="The file containing the training data",
                    type=argparse.FileType('r'),
                    required=True)
# parser.add_argument("-m", "--model",
#                     help="Output dir to store the regressor",
#                     type=argparse.FileType('wb'),
#                     required=True)
# parser.add_argument("-e", "--encoders",
#                     help="Output file to store the one hot encoders",
#                     type=argparse.FileType('wb'),
#                     required=True)
parser.add_argument("-p", "--plot",
                    help="Print graphs after training",
                    action="store_true")

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        # Load and preprocess the data
        print(f"Opening {args.input.name} ...")
        data = pd.read_csv(args.input)

        # Fill NAs
        data = data.fillna(0
        )
        # Preprocessing
        encoder = OneHotEncoder()
        scaler = StandardScaler()
        # X
        categoricals = data.loc[:, ['Best Hand']].values
        categoricals = [encoder.fit_transform(c) for c in categoricals]
        categoricals = np.concatenate(categoricals, axis=1).T
        numerical = data.iloc[:, 6:].values
        numerical = scaler.fit_transform(numerical)
        X = np.concatenate([categoricals, numerical], axis=1)
        # y
        y = data.loc[:, 'Hogwarts House'].values
    except Exception as e:
        print("Unable to open input dataset")
        raise e
        sys.exit(1)

    # Train the logistic regression
    clf = LogisticRegression()
    print("-"*10)
    print('Starting fitting process ...')
    clf.fit(X, y, verbose=True)
    print("-"*10)

    if args.plot:
        clf.plot()

    # # Save the model
    # try:
    #     pickle.dump(lr, args.model)
    # except Exception as e:
    #     print("Unable to save model")
    #     sys.exit(1)
