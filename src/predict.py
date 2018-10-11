#!/usr/bin/env python

import argparse
import csv
import pickle
import sys

import pandas as pd
import numpy as np

# Create CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="Path to the test data",
                    type=argparse.FileType('r'),
                    required=True)
parser.add_argument("-m", "--model",
                    help="Path to the trained model",
                    type=argparse.FileType('rb'),
                    required=True)
parser.add_argument("-e", "--encoder",
                    help="Path to the trained encoder",
                    type=argparse.FileType('rb'),
                    required=True)
parser.add_argument("-s", "--scaler",
                    help="Path to the trained scaler",
                    type=argparse.FileType('rb'),
                    required=True)
parser.add_argument("-o", "--output",
                    help="Path to save the predictions",
                    type=argparse.FileType('w'),
                    required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    modules = {
        'model': None,
        'encoder': None,
        'scaler': None
    }

    # Load the classifier, the encoder and the scaler
    for module in modules.keys():
        if hasattr(args, module):
            try:
                print(f'Opening {module} ({vars(args)[module].name}) ...')
                modules[module] = pickle.load(vars(args)[module])
            except Exception as e:
                print(f'Unable to open {module}')
                sys.exit(1)

    # Load and preprocess the data
    try:
        print(f"Opening {args.input.name} ...")
        data = pd.read_csv(args.input)

        # Fill NAs
        data = data.fillna(0)
        # X
        categoricals = data.loc[:, ['Best Hand']].values
        categoricals = [modules['encoder'].transform(c) for c in categoricals]
        categoricals = np.concatenate(categoricals, axis=1).T
        numerical = data.iloc[:, 6:].values
        numerical = modules['scaler'].transform(numerical)
        X = np.concatenate([categoricals, numerical], axis=1)
    except Exception as e:
        print("Unable to open and preprocess test dataset")
        sys.exit(1)

    # Predict
    try:
        preds = modules['model'].predict(X)
    except Exception as e:
        print("Unable to make predictions on the dataset")

    # Save predictions as CSV
    writer = csv.writer(args.output)
    writer.writerow(['Index', 'Hogwarts House'])
    for row in enumerate(preds):
        writer.writerow(row)
