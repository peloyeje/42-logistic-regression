#! /usr/bin/env python

import argparse
import pickle
import sys

import numpy as np

# Create CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
                    help="Trained model file",
                    type=argparse.FileType('rb'),
                    required=True)
parser.add_argument("-v", "--value",
                    help="Numerical mileage value to get price for",
                    type=float)

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the model
    try:
        print(f"Opening {args.model} ...")
        lr = pickle.load(args.model)
    except Exception as e:
        raise e
        print("Unable to open model")
        sys.exit(1)

    # Predict
    while True:
        # We enter an infinite loop to predict as many prices as the user wants
        # To exit, CTRL+C
        if args.value:
            value = args.value
            del args['value']
        else:
            value = float(input('Mileage (km) ? '))

        value = np.array([value])
        print(f"This car is estimated to be worth {lr.predict(value)[0]:.1f} euros")

