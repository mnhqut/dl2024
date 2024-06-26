import numpy as np
from gradient_descent import gradient_descent

class LinearRegression:
    def __init__(self, learning_rate=0.06, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.X = None
        self.y = None

    def loss_function(self,w):
        # Compute predictions
        predictions = self.X @ w.T

        # Compute error vector
        error = predictions - self.y
        return np.sum(error**2)/len(self.y)

    def fit(self, X, y):
        self.y = y
        # Add bias term (intercept) to X (a column with value 1)
        self.X = np.column_stack((np.ones(len(X)), X))

        # Initialize weights all 0
        init_weights = np.zeros(self.X.shape[1])

        # Perform gradient descent
        self.weights = gradient_descent(f = self.loss_function, initial_point = init_weights,
                                        learning_rate = self.learning_rate, max_iterations = self.max_iterations,
                                        tolerance = self.tolerance )

    def predict(self, X):
        # Add bias term (intercept) to X
        X = np.column_stack((np.ones(len(X)), X))
        # Predict
        return X @ self.weights.T