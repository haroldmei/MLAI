import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    preds = clf.predict(x_valid)
    #delta = preds - y_valid
    np.savetxt(pred_path, preds)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        i = 0
        while True:
            i += 1
            prev_theta = self.theta
            hx = np.exp(x.dot(self.theta))
            self.theta = self.theta + x.T.dot(y - hx) * self.step_size/m

            #test convergence
            delta = np.linalg.norm(prev_theta - self.theta)
            if delta < 1e-5:
                break
            
        print('theta = ', self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***


#debug one by one
if __name__ == '__main__':
    main(lr=1e-7,
         train_path='../data/ds4_train.csv',
         eval_path='../data/ds4_valid.csv',
         pred_path='output/p03d_pred.txt')
