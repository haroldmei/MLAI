import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(d): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given linear coefficients theta.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
