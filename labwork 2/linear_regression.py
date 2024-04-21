import numpy as np
from gradient_descent import gradient_descent
import matplotlib.pyplot as plt

def csv_to_numpy(file_path):
    
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return data

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.X = None
        self.y = None
    
    def loss_function(self,w):
        # Compute predictions
        predictions = np.dot(self.X, self.weights)
        
        # Compute error vector
        error = predictions - self.y
        return np.sum(error**2)/len(y)
    
    def fit(self, X, y):
        self.y = y
        # Add bias term (intercept) to X (a column with value 1)
        self.X = np.column_stack((np.ones(len(X)), X))
        
        # Initialize weights all 0
        init_weights = np.zeros(self.X.shape[1])
        
        # Perform gradient descent
        self.weights = gradient_descent(self.loss_function,init_weights)
        
    
    def compute_gradient(self, X, y):
        # Compute predictions
        predictions = np.dot(X, self.weights)
        
        # Compute error
        error = predictions - y
        
        # Compute gradient
        gradient = np.dot(X.T, error) / len(y)
        
        return gradient
    
    def predict(self, X):
        # Add bias term (intercept) to X
        X = np.column_stack((np.ones(len(X)), X))
        
        # Predict
        return np.dot(X, self.weights)

if __name__ == "__main__":
    data_array = csv_to_numpy('./data.csv')

    X = data_array[:, 0]  
    y = data_array[:, 1]
    #print(X.shape)
    #print(y.shape)
    lr = LinearRegression()
    lr.fit(X,y)
    print(lr.weights)
    pred = lr.predict(X)
    #print(pred)
    plt.plot(X, pred)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simple Line Plot')

    # Display the plot
    plt.show()