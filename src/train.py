#! /usr/bin/env python

import argparse
import pickle
import sys

from linear_regression import LinearRegression
import numpy as np

# Create CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="The file containing the training data",
                    type=argparse.FileType('r'),
                    required=True)
parser.add_argument("-m", "--model",
                    help="Output file to store the regressor",
                    type=argparse.FileType('wb'),
                    required=True)
parser.add_argument("-p", "--plot",
                    help="Print convergence and regression graphs",
                    action="store_true")

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the data
    try:
        print(f"Opening {args.input.name} ...")
        data = np.loadtxt(args.input, delimiter=',', skiprows=1)
        X = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print("Unable to open input dataset")
        sys.exit(1)

    # Fit the linear regression
    lr = LinearRegression()
    print("-"*10)
    print('Starting fitting process ...')
    weights, history = lr.fit(X, y)
    print("-"*10)
    print(f"Final weights: {weights}")
    print(f"MSE: {history[-1, 3]:.3f}")

    if args.plot:
        lr.plot_history()
        lr.plot_result(data)

    # Save the model
    try:
        pickle.dump(lr, args.model)
    except Exception as e:
        print("Unable to save model")
        sys.exit(1)
