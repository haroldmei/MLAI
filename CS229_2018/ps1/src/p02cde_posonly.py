import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # Answer to c:
    pred_path_c_plot = pred_path_c.replace('.', '_')
    # Train a logistic regression classifier and plot decision boundary on training set
    x_train, t_train = util.load_dataset(train_path, 't', add_intercept=True)
    lrc = LogisticRegression()
    lrc.fit(x_train, t_train)
    util.plot(x_train, t_train, lrc.theta, pred_path_c_plot + "_train.png")
    # Plot decision boundary on top of test set
    x_test, t_test = util.load_dataset(test_path, 't', add_intercept=False)
    util.plot(x_test, t_test, lrc.theta, pred_path_c_plot + "_test.png")
    # Use np.savetxt to save predictions on test set to pred_path_c
    probs, _ = lrc.predict(x_test)
    np.savetxt(pred_path_c, probs)
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    # answer to d:
    pred_path_d_plot = pred_path_d.replace('.', '_')
    _, y_train = util.load_dataset(train_path, add_intercept=True)
    lrd = LogisticRegression()
    lrd.fit(x_train, y_train)
    util.plot(x_train, t_train, lrd.theta, pred_path_d_plot + "_train.png")
    # Plot decision boundary on top of test set
    #x_test, y_test = util.load_dataset(test_path, add_intercept=False)
    util.plot(x_test, t_test, lrd.theta, pred_path_d_plot + "_test.png")
    # Use np.savetxt to save predictions on test set to pred_path_d
    probs, _ = lrd.predict(x_test)
    np.savetxt(pred_path_d, probs)


    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    # Answer to e:
    _, t_val = util.load_dataset(valid_path, 't', add_intercept=False)
    _, y_val = util.load_dataset(valid_path, add_intercept=False)
    alpha = np.sum(y_val) / np.sum(t_val)

    probs = probs / alpha
    np.savetxt(pred_path_e, probs)

    #y_test_pred = (probs > 0.5) + 0
    pred_path_e_plot = pred_path_e.replace('.', '_')
    util.plot(x_test, t_test, lrd.theta,
              pred_path_e_plot + "_test_pred.png", 0.2)


    # *** END CODER HERE



#debug one by one
if __name__ == '__main__':
    main(train_path='../data/ds3_train.csv',
        valid_path='../data/ds3_valid.csv',
        test_path='../data/ds3_test.csv',
        pred_path='output/p02X_pred.txt')
    
