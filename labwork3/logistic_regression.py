import numpy as np
from scipy.special import expit
from gradient_descent import gradient_descent

class LogisticRegression:
    def __init__(self, learning_rate=0.06, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.X = None
        self.y = None

    def loss_function(self,w):
        p = expit(self.X @ w.T)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        ones = np.ones(len(self.y))
        loss_vector = -np.log(p)*self.y - np.log(ones - p)* (ones-self.y)
        return np.sum(loss_vector)

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
        score = X @ self.weights.T
        pred = np.where(score < 0.5, 0, 1)
        return pred