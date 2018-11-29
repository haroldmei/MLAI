#!/usr/bin/env python

import numpy as np
import random
from p1_nn import *


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):

    #rndstate = random.getstate()
    #random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    grads = []
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        ### YOUR CODE HERE:
        # Use assignment will make a reference to x itself, we just want to make a copy of it
        xprime1 = x.copy()
        xprime1[ix] = x[ix] - h
        #random.setstate(rndstate)
        fxprime1, _ = f(xprime1)

        xprime2 = x.copy()
        xprime2[ix] = x[ix] + h
        #random.setstate(rndstate)
        fxprime2, _ = f(xprime2)

        numgrad = (fxprime2 - fxprime1) / (h * 2.0)

        grads.append(numgrad)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 5e-3:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print ("Gradient check passed!")


def fb_linear(flattened, labels, W2, b2, bias):
    logits = forward_linear(W2, b2, flattened)
    y = forward_softmax(logits)

    cost = forward_cross_entropy_loss(y, labels)

    grad_ce = backward_cross_entropy_loss(y, labels)
    grad_softmax = backward_softmax(logits, grad_ce)
    grad_linear = backward_linear(W2, b2, flattened, grad_softmax)
    
    if bias == True:
        return cost, grad_linear[1]
    else:
        return cost, grad_linear[0]
    # *** END CODE HERE ***

def fb_softmax(logits, labels):
    y = forward_softmax(logits)
    cost = forward_cross_entropy_loss(y, labels)
    grad_ce = backward_cross_entropy_loss(y, labels)
    grad_softmax = backward_softmax(logits, grad_ce)
    return cost, grad_softmax
    # *** END CODE HERE ***

def fb_ce(y, labels):
    cost = forward_cross_entropy_loss(y, labels)
    grad_ce = backward_cross_entropy_loss(y, labels)
    return cost, grad_ce
    # *** END CODE HERE ***

def fb_prop(data, labels, W1,b1,W2,b2, param):
    params = {}
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2

    grads = backward_prop(data,labels,params)
    if param == 'W1':
        return grads['cost'], grads['W1']
    elif param == 'b1':
        return grads['cost'], grads['b1']
    elif param == 'W2':
        return grads['cost'], grads['W2']
    elif param == 'b2':
        return grads['cost'], grads['b2']


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print ("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ("")


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")

    ### YOUR CODE HERE
    labels = np.zeros(10)
    labels[np.random.randint(0, 10)] = 1

    print ("==== Gradient check for ce ====")
    yhat = np.random.dirichlet(np.ones(10))
    gradcheck_naive(lambda y: fb_ce(y, labels), yhat)
    
    print ("==== Gradient check for softmax ====")
    logits = np.random.rand(10)*10
    gradcheck_naive(lambda lgt: fb_softmax(lgt, labels), logits)

    params = get_initial_params()
    flattened = np.random.rand(50)
    print ("==== Gradient check for linear W2====")
    gradcheck_naive(lambda W2: fb_linear(flattened, labels, W2, params['b2'], False), params['W2'])

    print ("==== Gradient check for linear b2====")
    gradcheck_naive(lambda b2: fb_linear(flattened, labels, params['W2'], b2, True), params['b2'])

    test_data, test_labels = read_data('../data/images_test.csv', '../data/labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    data, labels = test_data[1], test_labels[1]
    W1, b1, W2, b2 = params['W1'],params['b1'],params['W2'],params['b2']
    print ("==== Gradient check for all W1====")
    gradcheck_naive(lambda W1: fb_prop(data, labels, W1, b1, W2, b2, 'W1'), params['W1'])

    print ("==== Gradient check for all b1====")
    gradcheck_naive(lambda b1: fb_prop(data, labels, W1, b1, W2, b2, 'b1'), params['b1'])

    print ("==== Gradient check for all W2====")
    gradcheck_naive(lambda W2: fb_prop(data, labels, W1, b1, W2, b2, 'W2'), params['W2'])

    print ("==== Gradient check for all b2====")
    gradcheck_naive(lambda b2: fb_prop(data, labels, W1, b1, W2, b2, 'b2'), params['b2'])

    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
