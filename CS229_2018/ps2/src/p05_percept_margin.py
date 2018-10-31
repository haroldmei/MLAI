import math

import matplotlib.pyplot as plt
import numpy as np

import util


def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron.

    """

    # *** START CODE HERE ***
    return ([],[],[],[])
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Peform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance
    
    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    X = np.array(state[0]).T
    kern = state[3]
    #kernMatrix = X.T.dot(X)
    
    #num = len(state[0])
    #kernMatrix = np.zeros([num,num])
    #for i in range(num):
    #    for j in range(num):
    #        kernMatrix[i][j] = kernel(np.array(state[0][i]),np.array(state[0][j]))
        
    beta = np.linalg.solve(kern, X.T.dot(x_i))
    
    X_theta = np.array(state[2]).T[-1]
    inner = beta.dot(X_theta)
    return sign(inner)
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    # Append to the last column
    length = len(state[0])

    # add a new row
    state[0].append(x_i)
    state[1].append(y_i)
    state[2].append([0.0])
    state[3].append([])
    index = 0 
    while index < length + 1:
        kernEnt = kernel(state[0][index], x_i)
        oldMargin = state[2][length][index] 
        state[2][length].append(oldMargin + learning_rate * (state[1][index] - sign(state[2][index][index])) * kernEnt)
        state[3][length].append(kernEnt)
        index = index + 1

    index1 = 0 
    while index1 < length:
        kernEnt = kernel(x_i, state[0][index1])
        oldMargin = state[2][index1][length] 
        #state[2][index] = oldMargin + learning_rate * (y_i - sign(oldMargin)) * kernEnt
        state[2][index1].append(oldMargin + learning_rate * (y_i - sign(state[2][length][length])) * kernEnt)
        state[3][index1].append(kernEnt)
        index1 = index1 + 1
    # *** END CODE HERE ***


def sign(a):
    """Gets the sign of a scalar input."""
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)


def train_perceptron(kernel_name, kernel, learning_rate):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/output/p05_{kernel_name}_predictions.txt.
    The output plots are saved to src/output_{kernel_name}_output.pdf.

    Args:
        kernel_name: The name of the kernel.
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    train_x, train_y = util.load_csv('../data/ds5_train.csv')

    state = initial_state()

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv('../data/ds5_train.csv')

    result = []
    for i in range(len(test_x)):
        est_y = predict(state, kernel, test_x[i])
        result.append((est_y==test_y[i])+0.)
    
    #print(test_y)
    #print(np.array(result))

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.savefig('./output/p05_{}_output.pdf'.format(kernel_name))

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('./output/p05_{}_predictions'.format(kernel_name), predict_y)


def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)


if __name__ == "__main__":
    main()
