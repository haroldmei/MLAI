import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    util.plot(x_train, y_train, lr.theta, pred_path + "_train.png")

    # Plot decision boundary on top of validation set set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    util.plot(x_valid, y_valid, lr.theta, pred_path + "_valid.png")

    # Use np.savetxt to save predictions on eval set to pred_path
    probs,_ = lr.predict(x_valid)
    np.savetxt(pred_path, probs)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Logistic regression model parameters, including intercept.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        i = 0
        while True:
            i += 1
            prev_theta = self.theta
            hx = 1. / (1 + np.exp(x.dot(self.theta)))

            #hessian d2J/dtheta2 = mean(h(x) * (1 - h(x)) * x * x.T)
            H1 = (1./m) * hx * (1 - hx) * x.T   
            H = H1.dot(x)

            #gradient dJ/dtheta = mean((h(x) - y) * x)
            gradient = -(1./m) * x.T.dot(y - hx)

            #update theta
            self.theta = self.theta + np.linalg.inv(H).dot(gradient)

            #test convergence
            if np.linalg.norm(prev_theta - self.theta) < 1e-5:
                #print("Converged in %d iterations"%i)
                break

        #print('theta = ', self.theta)
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
        new_x = util.add_intercept(x)
        probs = 1. / (1 + np.exp(new_x.dot(self.theta)))
        clss = ((probs > 0.5) + 0)  # convert to {0,1}
        return probs, clss
        # *** END CODE HERE ***

#debug one by one
if __name__ == '__main__':
    main(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01b_pred_1.txt')
    
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')