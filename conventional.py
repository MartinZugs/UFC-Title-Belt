import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

if __name__ == '__main__':
    # path to the csv file
    DATA_PATH = "manually-preprocessed_data-full-headers.csv"
    # number of folds
    K_FOLDS = 10
    # maximum iterations
    MAX_ITER = 20000
    # random state to use for shuffling
    RANDOM_STATE = 987654321
    # multi-core threads
    MULTI_CORE = -1

    # import data
    pd_data = pd.read_csv(DATA_PATH)

    # x and y data
    pd_data_y = pd_data["Winner"]
    pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]

    ####################################################################################################################
    print()
    print("RUNNING PERCEPTRON")
    # parameters to search over
    per_parameters = {'penalty': ('None', 'l1', 'l2', 'elasticnet'),
                      'warm_start': ('True', 'False'),
                      'class_weight': (None, 'balanced')
                      } #   'eta0': ('1', '.1', '.01'), 'fit_intercept': ('True', 'False')

    # logistic regression definition
    per = Perceptron(shuffle=True, random_state=RANDOM_STATE, max_iter=MAX_ITER, verbose=10) # n_jobs=MULTI_CORE
    # gridsearch over the parameters
    clf = GridSearchCV(per, per_parameters, cv=K_FOLDS, n_jobs=MULTI_CORE, verbose=10)
    # fit the date
    clf.fit(pd_data_x, pd_data_y)

    # result = sorted(clf.cv_results_.keys())
    print()
    print("PERCEPTRON RESULTS:")
    print(clf.cv_results_["mean_test_score"])

    ####################################################################################################################
    print()
    print("RUNNING LOGISTIC REGRESSION")
    # parameters to search over
    log_parameters = {'solver': ('lbfgs', 'liblinear'), 'C': [1, 10]}
    # logistic regression definition
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER)
    # gridsearch over the parameters
    clf = GridSearchCV(log_reg, log_parameters, cv=K_FOLDS, n_jobs=MULTI_CORE, verbose=10)
    # fit the date
    clf.fit(pd_data_x, pd_data_y)

    # result = sorted(clf.cv_results_.keys())
    print()
    print("LOGISTIC REGRESSION RESULTS:")
    print(clf.cv_results_["mean_test_score"])
