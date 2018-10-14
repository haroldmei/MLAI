import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    num_test = len(tau_values)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    min_mse = 10    # use a large number as long as it's larger than max(y)
    best_tau = tau_values[0]
    for i in range(num_test):
        # Fit a LWR model
        clf = LocallyWeightedLinearRegression(tau_values[i])
        clf.fit(x_train, y_train)

        # Get MSE value on the validation set
        y_pred = clf.predict(x_valid)
        err = y_valid - y_pred
        mse = np.mean(err * err)
        plt.figure()
        clf.plot(plt, x_train, y_train, x_valid, y_pred, "output/p05c_pred_%d.png" % i)
        print("PS5.c mean square error for tau=%f is: %f" % (tau_values[i], mse))
        if mse < min_mse:
            min_mse=mse
            best_tau = tau_values[i]
        
    print("PS5.c min mse is: %f, best tau is %f" % (min_mse, best_tau))
            

    # Fit a LWR model with the best tau value
    best_lwr = LocallyWeightedLinearRegression(best_tau)
    best_lwr.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = best_lwr.predict(x_test)
    err = y_test - y_pred
    mse_test = np.mean(err * err)
    print("PS5.c mean square error for test set is: %f" % mse_test)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plt.figure()
    clf.plot(plt, x_train, y_train, x_test, y_pred, "output/p05c_test.png")
    # *** END CODE HERE ***

#debug one by one
if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='../data/ds5_train.csv',
         valid_path='../data/ds5_valid.csv',
         test_path='../data/ds5_test.csv',
         pred_path='output/p05c_pred.txt')