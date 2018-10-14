import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    pred_path_plot = pred_path.replace('.', '_')
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_train, y_train, gda.theta, pred_path_plot + "_gda_train.png")

    # Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_valid, y_valid, gda.theta, pred_path_plot + "_gda_valid.png")

    # Use np.savetxt to save outputs from validation set to pred_path
    probs,_ = gda.predict(x_valid)
    np.savetxt(pred_path, probs)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)    # GDA input without intercepts
        num_true = np.sum(y)
        # Find phi, mu_0, mu_1, and sigma
        phi = num_true / m
        mu_0 = x.T.dot(1 - y) / (m - num_true)
        mu_1 = x.T.dot(y) / num_true
        mu = np.zeros((m, n))
        mu[y==1] = mu_1
        mu[y==0] = mu_0
        delta = x - mu
        sigma = delta.T.dot(delta) / m

        # Write theta in terms of the parameters
        # theta = (u_0 - u_1) * sigma^{-1}
        # theta0 = log[(1-phi)/phi] - (u_0^T inv(sigma) u_0 - u_1^T inv(sigma) u_1)
        factor = np.log(1 - phi) - np.log(phi)
        sigma_inv = np.linalg.inv(sigma)
        self.theta[1:] = sigma_inv.dot(mu_0 - mu_1)
        b0 = mu_0.dot(sigma_inv).dot(mu_0)
        b1 = mu_1.dot(sigma_inv).dot(mu_1)
        self.theta[0] = factor - (b0 - b1)/2

        print('theta = ', self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        new_x = util.add_intercept(x)
        probs = 1. / (1 + np.exp(new_x.dot(self.theta)))
        clss = ((probs > 0.5) + 0)  # convert to {0,1}
        return probs, clss
        # *** END CODE HERE



#debug one by one
if __name__ == '__main__':
    main(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01e_pred_1.txt')

    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01e_pred_2.txt')