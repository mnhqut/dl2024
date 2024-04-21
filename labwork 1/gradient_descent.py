import numpy as np

# Define a function to compute the gradient
# input: multivariate function that range be R; x: position want to calc gradient,represented as a vector; h: parameter to estimate 
# output: gradient vector at that position
def compute_gradient(f, x, h=1e-6):
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        gradient[i] = (f(x_plus_h) - f(x_minus_h)) / (2*h)
    return gradient


def gradient_descent(f, initial_point, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
    print('(time, x, f(x)):')
    x = initial_point
    for i in range(max_iterations):
        gradient = compute_gradient(f, x)
        x -= learning_rate * gradient
        print(i,x,f(x))
        if np.linalg.norm(gradient) < tolerance:
            print(f"Gradient descent converged at iteration {i+1}")
            break
    else:
        print("Gradient descent did not converge within the maximum number of iterations.")
    return x
        
if __name__ == "__main__":
    # Define the multivariate function
    def function(x):
        return x[0]**2-4*x[0] +5
    
    x = gradient_descent(function,[5])