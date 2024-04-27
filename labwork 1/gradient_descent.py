import numpy as np

# Define a function to compute the approximate gradient
# input: multivariate function that has range R; x: position want to calc gradient,represented as a vector; h: parameter to estimate
# output: approximated gradient vector at that position
def compute_gradient(f, x, h=1e-6):
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        gradient[i] = (f(x_plus_h) - f(x_minus_h)) / (2*h)
    return gradient

# input: 'f', 'initial_point', 'learning_rate', 'max_iterations', 'tolerance'
# output: argument that minimize f, the values of arguments of f at each iteration and the value of f at each iteration
def gradient_descent(**kwargs):
    valid_args = {'f', 'initial_point', 'learning_rate', 'max_iterations', 'tolerance'}

    # Check for invalid keyword arguments
    invalid_args = set(kwargs.keys()) - valid_args
    if invalid_args:
        raise ValueError(f"Invalid keyword arguments: {', '.join(invalid_args)}")

    # Check if required keyword arguments are provided
    if 'f' not in kwargs:
        raise ValueError("Function 'f' must be provided.")
    if 'initial_point' not in kwargs:
        raise ValueError("Initial point must be provided.")

    # Extracting keyword arguments with default values
    f = kwargs.get('f')
    initial_point = kwargs.get('initial_point')
    learning_rate = kwargs.get('learning_rate', 0.05)
    max_iterations = kwargs.get('max_iterations', 1000)
    tolerance = kwargs.get('tolerance', 1e-6)

    print('(time, x, f(x)):')
    x = initial_point
    x_list = [np.copy(x)]
    func_list = [f(x)]
    for i in range(max_iterations):
        print(i,x,f(x))
        gradient = compute_gradient(f, x)
        x -= learning_rate * gradient
        x_list.append(np.copy(x))
        func_list.append(f(x))
        if np.linalg.norm(gradient) < tolerance:
            print(f"Gradient descent converged at iteration {i+1}")
            break
    else:
        print("Gradient descent did not converge within the maximum number of iterations.")
    return x, x_list, func_list
