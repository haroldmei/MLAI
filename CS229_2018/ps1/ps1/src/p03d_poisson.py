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
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Logistic regression model parameters, including intercept.
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given logistic
        regression parameters theta. Input will not have an intercept term
        (i.e. not necessarily x[0] = 1), but theta expects an intercept term.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
