import random
import csv
import numpy as np
import math
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def GenerateData(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])
    N = data.shape[0]
    X = data[:, :-1].astype(np.float64)
    X = normalize(X, axis=0, norm="max")  # avoid overflow problem

    y = data[:, -1].astype(np.int32).reshape(N, 1)
    test_id = np.random.choice(N, int(0.3 * N), replace=False)
    X_train = np.delete(X, test_id, axis=0)
    y_train = np.delete(y, test_id, axis=0)
    X_test = X[test_id]
    y_test = y[test_id]
    print("Dimension of the feature data in training set:", X_train.shape)
    print("Dimension of the feature data in testing set:", X_test.shape)
    return X_train, y_train, X_test, y_test



def PlotROCs(y_prob_lr, y_prob_tree, y_prob_svm, y_test):
    # %%
    # TODO: Plot the ROCs in one graph
    pass
    # %%


def main(data_path):
    (X_train, y_train, X_test, y_test,) = GenerateData(data_path)
    # %%
    # TODO: Plot the ROCs in one graph

    lr_model = None
    tree_model = None
    svm_model = None

    # predicted probability on test set using logistic regression model
    y_prob_lr = None 

    # predicted probability on test set using tree model
    y_prob_tree = None

    # predicted probability on test set using svm model
    y_prob_svm = None
    # %%


    PlotROCs(y_prob_lr, y_prob_tree, y_prob_svm, y_test)

if __name__ == "__main__":
    set_seed(6)
    data_path = "data.csv"
    main(data_path)


