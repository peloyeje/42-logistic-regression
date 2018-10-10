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
    def _model(X, b):
        """Basic linear regression model"""
        return np.dot(X, b)

    @staticmethod
    def _loss(y_true, y_pred):
        """Computes the MSE"""
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Length mismatch between ground truth and prediction vector')
        return np.mean((y_true - y_pred) ** 2)

    def _scale(self, X):
        """Normalize data"""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return (X - self.mean) / self.std

    def _intercept(self, X):
        """Add intercept (column of ones) to the design matrix X"""
        return np.c_[np.ones(X.shape[0]), X]

    def _descale_beta(self, b):
        """Denormalize weights"""
        b = b.copy()
        b[0] = (b[0] * self.std[1] - b[1] * self.mean[0] * (self.std[1] / self.std[0])) + self.mean[1]
        b[1] = b[1] * (self.std[1] / self.std[0])
        return b

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

        # Scale dataset
        data = self._scale(np.c_[X, y])
        X, y = data[:, :-1], data[:, -1]
        # Add intercept column
        X = self._intercept(X)
        # Initialize weight vector
        self.beta = np.ones(X.shape[1])
        self.history = np.zeros((self.max_iterations, 4))

        # Iterate until we reach convergence or maximum number of iterations
        for i in range(self.max_iterations):
            # Compute gradient
            error = self._model(X, self.beta) - y
            gradient = (2.0/X.shape[0]) * np.dot(X.T, error)

            # Update beta according to learning rate
            beta = self.beta - (self.lr * gradient)
            loss = self._loss(
                self._model(X, beta),
                y
            )

            # Store history
            self.history[i, :] = (i, *self._descale_beta(self.beta), loss)

            if np.sum(np.abs(beta - self.beta)) < self.threshold:
                # We reached sufficient precision, let's exit the loop
                print(f'Convergence reached in {i} iterations, exiting the loop ...')
                break
            else:
                # We store the new weights
                self.beta = beta
                self.loss = loss

            # Print info on the current iteration if needed
            if verbose and i % 10 == 0:
                print(
                    f'[{i:5}] Loss: {self.loss:20} | Beta: {self._descale_beta(self.beta)}')

        # When convergence is reached, denormalize the beta vector and store it
        self.beta = self._descale_beta(self.beta)

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
