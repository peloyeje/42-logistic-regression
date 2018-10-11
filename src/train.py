#! /usr/bin/env python

import argparse
import pathlib
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
parser.add_argument("-o", "--output",
                    help="Output dir to store the classifier and the encoders",
                    type=pathlib.Path,
                    required=True)
parser.add_argument("-p", "--plot",
                    help="Print graphs after training",
                    action="store_true",
                    default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.output.is_dir() or not args.output.exists():
        raise ValueError(f'{args.output} is not a valid directory')

    # Load and preprocess the data
    try:
        print(f"Opening {args.input.name} ...")
        data = pd.read_csv(args.input)

        # Fill NAs
        data = data.fillna(0)
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
        sys.exit(1)

    # Train the logistic regression
    clf = LogisticRegression()
    print("-"*10)
    print('Starting fitting process ...')
    clf.fit(X, y, verbose=True)
    print("-"*10)

    if args.plot:
        clf.plot()

    # Save the classfier, encoder and scaler
    to_save = {
        'model': clf,
        'encoder': encoder,
        'scaler': scaler
    }

    for name, obj in to_save.items():
        try:
            path = args.output / name

            print(f'Saving {name} to {path.absolute()} ...')
            with path.open('wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            print(f'Unable to save {name}')
            sys.exit(1)
