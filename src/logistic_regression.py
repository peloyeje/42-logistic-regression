import numpy as np
from matplotlib import pyplot as plt

from .utils import OneHotEncoder

class LogisticRegression:
    SUPPORTED_ALGORITHMS = ('gd', 'sgd')

    def __init__(
        self, algorithm='gd', multiclass='ovr', lr=0.01,
        max_iterations=1000, threshold=1e-3):
        """Initialize logistic regression model

        Parameters
        ----------
        algorithm: str
            Training algorithm ('gd' or 'sgd')
        multiclass: str
            The way to handle multiclass targets ('ovr' or None)
        lr: float
            Learning rate
        max_iterations: int
            Maximum number of GD iterations
        threshold: float
            If the difference between the sum of the parameters at iteration
            t+1 and at iteration t is lower than this value, we assume
            convergence has been reached

        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f'Algorithm "{algorithm}" not supported')
        else:
            self.algorithm = algorithm

        if multiclass not in ('ovr', None):
            raise ValueError(f'Multiclass method "{algorithm}" not supported')
        else:
            self.multiclass = multiclass

        self.lr = float(lr)
        self.max_iterations = int(max_iterations)
        self.threshold = float(threshold)

        self._models = []
        self._histories = []

    @staticmethod
    def _sigmoid(x):
        """Link function between Xb and y"""
        return 1 / (1 + np.exp(-x))

    @classmethod
    def _model(cls, X, b):
        """Logistic regression model"""
        return cls._sigmoid(np.dot(X, b))

    @classmethod
    def _gradient(cls, X, b, y):
        """Gradient of the log likelihood formula"""
        return np.dot(X.T, (y - cls._model(X, b)))

    @classmethod
    def _log_likelihood(cls, X, b, y):
        """Log likelihood of the parameters"""
        return np.mean(
            y * np.log(cls._model(X, b)) + (1 - y) * np.log(1 - cls._model(X, b))
        )

    @staticmethod
    def _accuracy_score(y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError(
                'Prediction and ground truth vectors must have the same dimension')
        return (y_pred == y_true).sum() / y_pred.size

    @staticmethod
    def _intercept(X):
        """Add intercept (column of ones) to the design matrix X"""
        return np.c_[np.ones(X.shape[0]), X]

    @property
    def is_multiclass(self):
        return len(self._models) > 0

    def fit(self, X, y, **kwargs):
        """
        """
        if len(np.unique(y)) > 2:
            # There are more than 2 classes in the target column
            if self.multiclass == 'ovr':
                # One Hot Encode the target column
                self._encoder = OneHotEncoder()
                y = self._encoder.fit_transform(y)
                # Train one model per class
                for name, data in zip(self._encoder.categories, y.T):
                    print(f'Training model for class "{name}"')

                    model = LogisticRegression()
                    _, history = model._fit(X, data, verbose=False, **kwargs)
                    # Store trained models and histories in parent class
                    self._models.append(model)
                    self._histories.append(history)
        else:
            # We directly call the underlying method
            self._fit(X, y, **kwargs)


    def _fit(self, X, y, verbose=True):
        """Trains a logistic regression model

        Parameters
        ----------
        X: np.array
            Training data
        y: np.array
            Target

        Returns
        -------
        beta: np.array
            Vector of model weights

        history: np.array
            Performance history of the convergence process
            Each row contains : the beta parameters and the loss

        """
        # Add intercept column
        X = self._intercept(X)

        # Initialize weight vector
        self.beta = np.ones(X.shape[1])
        self.history = np.zeros((self.max_iterations, 3))

        # Iterate until we reach convergence or maximum number of iterations
        for i in range(self.max_iterations):
            # We save n-1 beta for convergence test
            beta = self.beta

            # Compute gradient and update weights according to learning rate
            self.beta = self.beta + (self.lr * self._gradient(X, self.beta, y))
            self.log_likelihood = self._log_likelihood(X, self.beta, y)
            self.accuracy = self._accuracy_score(self._model(X, self.beta) > .5, y)

            # Store history
            self.history[i, :] = (i, self.log_likelihood, self.accuracy)

            # If we reached sufficient precision, let's exit the loop
            if np.sum(np.abs(beta - self.beta)) < self.threshold:
                print(f'Convergence reached in {i} iterations, exiting the loop ...')
                break

            # Print info on the current iteration if needed
            if verbose and i % 10 == 0:
                print(
                    f'[{i:5}] Train accuracy: {self.accuracy:10.3%} | LL: {self.log_likelihood:.4f}')

        return self.beta, self.history

    def predict_proba(self, X):
        """Predicts target probability according to input data"""
        if self.is_multiclass:
            return np.array([m.predict_proba(X) for m in self._models]).T
        else:
            X = self._intercept(X)
            return self._model(X, self.beta)

    def predict(self, X, threshold=0.5):
        """Returns class predictions"""
        if self.is_multiclass:
            return self.predict_proba(X).argmax(axis=1)
        else:
            return self.predict_proba(X) >= threshold

    def plot_history(self):
        """Plots a summary graph of the fitting process."""
        if not hasattr(self, 'history'):
            raise ValueError('Please train the model first')

        labels = ['Train accurary', 'LL']
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        for ax, data, label in zip(axes, self.history[self.history[:, 0] != 0, 1:].T, labels):
            ax.plot(np.arange(data.size), data)
            ax.legend([label])
            ax.set_xlabel('Iterations')
        fig.suptitle(f'Logistic regression on {len(self.beta) - 1} features')
        plt.show()
