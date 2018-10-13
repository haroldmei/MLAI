import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_valid)
    err = y_valid - y_pred
    mse = np.mean(err * err)
    print("PS5.c mean square error is: %f" % mse)
    # Plot validation predictions on top of training set
    plt.figure()
    clf.plot(plt, x_train, y_train, x_valid, y_pred, "lwr_pred.png")

    # No need to save predictions
    # Plot data
    plt.figure()
    clf.plot(plt, x_train, y_train, x_valid, y_valid, "lwr_data.png")
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,_ = x.shape
        yhat = np.zeros(m)
        denom = 2. * self.tau ** 2
        i = 0
        while i < m:
            xx = self.x - x[i]
            w = np.exp([-x.dot(x)/denom for x in xx]) #any idea?
            W = np.diag(w)
            A = self.x.T.dot(W)
            theta = np.linalg.inv(A.dot(self.x)).dot(A).dot(self.y)
            yhat[i] = x[i].dot(theta)
            i += 1

        return yhat
        # *** END CODE HERE ***

    def plot(self, y_pred, x_train, y_train, x_valid, y_valid, save_path):
        """Plot dataset"""
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_valid, y_valid, 'ro', linewidth=2)
        
        # Add labels and save to disk
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(save_path)

#debug one by one
if __name__ == '__main__':
    main(tau=5e-1,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')