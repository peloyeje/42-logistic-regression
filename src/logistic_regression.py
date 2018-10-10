import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, max_iterations=1000, threshold=1e-4):
        """Initialize logistic regression model

        Parameters
        ----------
        lr: float
            Learning rate
        max_iterations: int
            Maximum number of GD iterations
        threshold: float
            If the difference between the sum of the parameters at iteration
            t+1 and at iteration t is lower than this value, we assume
            convergence has been reached
        """

        self.lr = float(lr)
        self.max_iterations = int(max_iterations)
        self.threshold = float(threshold)

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

    def fit(self, X, y, verbose=True):
        """Learns the weights of the regression model

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

            if np.sum(np.abs(beta - self.beta)) < self.threshold:
                # We reached sufficient precision, let's exit the loop
                print(f'Convergence reached in {i} iterations, exiting the loop ...')
                break

            # Print info on the current iteration if needed
            if verbose and i % 10 == 0:
                print(
                    f'[{i:5}] Accuracy: {self.accuracy:10.3%} | LL: {self.log_likelihood:.4f}')

        return self.beta, self.history

    def plot_history(self):
        """Plots a summary graph of the fitting process."""

        if not hasattr(self, 'history'):
            raise ValueError('Please train the model first')

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        labels = ['Intercept', 'Slope', 'Loss']

        for ax, data, label in zip(axes, self.history[self.history[:, 0] != 0, 1:].T, labels):
            ax.plot(np.arange(data.size), data)
            ax.legend([label])
            ax.set_xlabel('Iterations')
        fig.suptitle(f'Weights: {self.beta} | Loss: {self.loss}')
        plt.show()

    def plot_result(self, X):
        """Plots the regression line alongside data points"""

        if not hasattr(self, 'history'):
            raise ValueError('Please train the model first')

        fig, ax = plt.subplots(figsize=(10, 5))
        projection = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        ax.scatter(X[:, 0], X[:, 1], label='Data', color='b')
        ax.plot(projection, self.predict(projection), label='Regression', color='r')

        fig.suptitle(f'Actual data vs. projected data')
        plt.show()

    def predict(self, X):
        """Predicts target according to input data"""
        X = self._intercept(X)
        return self._model(X, self.beta)
