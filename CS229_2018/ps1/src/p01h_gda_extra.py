import numpy as np
import util

from p01e_gda import GDA



def main_extra(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    pred_path_plot = pred_path.replace('.', '_')

    # *** START CODE HERE ***
    # square root
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    x_train[:,1] = np.abs(x_train[:,1]) ** 0.5
    x_valid[:,1] = np.abs(x_valid[:,1]) ** 0.5
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, pred_path_plot + "_gda_valid_extra_2.png")

    # 4th root
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    x_train[:,1] = np.abs(x_train[:,1]) ** 0.25
    x_valid[:,1] = np.abs(x_valid[:,1]) ** 0.25
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, pred_path_plot + "_gda_valid_extra_4.png")

    # log
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    x_train[:,1] = np.log(x_train[:,1])
    x_valid[:,1] = np.log(x_valid[:,1])
    gda = GDA()
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, pred_path_plot + "_gda_valid_extra_log.png")
    # *** END CODE HERE ***


#debug one by one
if __name__ == '__main__':
    main_extra(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01e_pred_1.txt')