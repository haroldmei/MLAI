from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from numpy.linalg import inv

def load_data():
    X = np.genfromtxt('logistic_x.txt')
    Y = np.genfromtxt('logistic_y.txt')
    return X, Y

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))

    ################

    X[:, 0] = 1
    X[:, 1:] = X_

    ################

    return X

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    ##############
    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * X.T.dot(Y * probs)
    ##############

    return grad

##
## This function is useful to debug
## Ensure that loss is going down over iterations
##
def calc_loss(X, Y, theta):
    m, n = X.shape
    loss = 0.

    ###########
    margins = Y * X.dot(theta)
    probs = np.log(1. / (1 + np.exp(-margins)))
    ONES = np.ones((m, 1))
    loss = -(1./m) * probs.dot(ONES)
    ###########

    return loss

def calc_hessian(X, Y, theta):
    m, n = X.shape
    H = np.zeros((n, n))

    ##############
    margins = Y * X.dot(theta)
    probs1 = 1. / (1 + np.exp(margins)) #this is g(z)
    probs2 = 1. / (1 + np.exp(-margins))#this is (1 - g(z))
    probs = -(1./m) * probs1 * probs2

    H1 = probs * X.T

    H = H1.dot(X)
    #############

    return H
    
def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)

    i = 0
    while True:
        i += 1
        prev_theta = theta
        ############
        H = calc_hessian(X, Y, theta)
        HI = inv(H)
        G = calc_grad(X,Y,theta)
        theta = theta + HI.dot(G)
        ############
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break

    print('theta = ', theta)
    return theta
    
def plot(X, Y, theta):
    plt.figure()

    ############
    ## Plot Examples
    pos = X[np.where(Y==1,True,False).flatten()]
    neg = X[np.where(Y==-1,True,False).flatten()]
    plt.plot(pos[:,1], pos[:,2], '+', markersize=7, markeredgecolor='black', markeredgewidth=2)
    plt.plot(neg[:,1], neg[:,2], 'o', markersize=7, markeredgecolor='black', markerfacecolor='yellow')

    ## Plot Boundary
    plot_x = np.array([min(X[:, 1]),  max(X[:, 1])])
    plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
    plt.plot(plot_x, plot_y)
    ############

    plt.savefig('ps1q1c.png')
    return

def main():
    X_, Y = load_data()
    X = add_intercept(X_)
    theta = logistic_regression(X, Y)
    plot(X, Y, theta)

if __name__ == '__main__':
    main()
