import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import time
import re
import sys

if __name__ == '__main__':
    # path to the csv file
    DATA_PATH = "manually-preprocessed_data-full-headers.csv" #'preprocessed_data_condensed.csv' # "manually-preprocessed_data-full-headers.csv"
    # number of folds
    K_FOLDS = 10
    # maximum iterations
    MAX_ITER = 20000
    # random state to use for shuffling
    RANDOM_STATE = 987654321
    # multi-core threads. -1
    MULTI_CORE = -1
    # VERBOSITY
    VERBOSE = 1

    # import data
    pd_data = pd.read_csv(DATA_PATH)

    # x and y data
    pd_data_y = pd_data["Winner"]
    pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]



    ####################################################################################################################
    print()
    print("RUNNING SVM")
    # parameters to search over
    svm_parameters = {'C': [.000001,.00001,.0001, .001,.01, .1],  # np.arange(.01, 10, .1)
                      'gamma': [.000001,.00001,.0001, .001,.01, .1]} # np.arange(.01, 10, .01)
                      #'gamma':('auto', 'scale')} # 'kernel':'rbf', #, 'linear', 'sigmoid'

    # logistic regression definition
    svm = SVC(random_state=RANDOM_STATE, verbose=VERBOSE)  # n_jobs=MULTI_CORE ,max_iter=MAX_ITER
    # gridsearch over the parameters
    clf = GridSearchCV(svm, svm_parameters, cv=K_FOLDS, n_jobs=MULTI_CORE, verbose=VERBOSE)
    # fit the date
    clf.fit(pd_data_x, pd_data_y)

    best_score_svm = clf.best_score_
    best_params_svm = clf.best_params_

    print("")
    print("")
    print("SVM RESULTS:")
    print("BEST SCORE: ", best_score_svm)
    print("BEST PARAMS: ", best_params_svm)

    # wait for 10 seconds so the user can see the score!
    time.sleep(10)

    ####################################################################################################################
    print()
    print("RUNNING PERCEPTRON")
    # parameters to search over
    per_parameters = {'penalty': ('None', 'l1', 'l2', 'elasticnet'),
                      'warm_start': ('True', 'False'),
                      'class_weight': (None, 'balanced')
                      }  # 'eta0': ('1', '.1', '.01'), 'fit_intercept': ('True', 'False')

    # logistic regression definition
    per = Perceptron(shuffle=True, random_state=RANDOM_STATE, max_iter=MAX_ITER, verbose=VERBOSE)  # n_jobs=MULTI_CORE
    # gridsearch over the parameters
    clf = GridSearchCV(per, per_parameters, cv=K_FOLDS, n_jobs=MULTI_CORE, verbose=VERBOSE)
    # fit the date
    clf.fit(pd_data_x, pd_data_y)

    best_score_perceptron = clf.best_score_
    best_params_perceptron = clf.best_params_

    print("")
    print("")
    print("PERCEPTRON RESULTS:")
    print("BEST SCORE: ", best_score_perceptron)
    print("BEST PARAMS: ", best_params_perceptron)

    # wait for 10 seconds so the user can see the score!
    time.sleep(10)

    ####################################################################################################################
    print()
    print("RUNNING LOGISTIC REGRESSION")
    # parameters to search over
    log_parameters = {'penalty': ('none', 'l2'),
                      'warm_start': ('True', 'False'),
                      'class_weight': (None, 'balanced'),
                      'C': np.arange(4, 5, .01)} # 'C': [8,9,9.5,10,10.5,11,12]

    # logistic regression definition
    log_reg = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE, n_jobs=MULTI_CORE, max_iter=MAX_ITER, verbose=VERBOSE)
    # gridsearch over the parameters
    clf = GridSearchCV(log_reg, log_parameters, cv=K_FOLDS, n_jobs=MULTI_CORE, verbose=VERBOSE)
    # fit the date
    clf.fit(pd_data_x, pd_data_y)

    best_score_log = clf.best_score_
    best_params_log = clf.best_params_

    # wait for 10 seconds so the user can see the score!
    time.sleep(10)


    print("")
    print("")
    print("SVM RESULTS:")
    print("BEST SCORE: ", best_score_svm)
    print("BEST PARAMS: ", best_params_svm)

    print("")
    print("")
    print("PERCEPTRON RESULTS:")
    print("BEST SCORE: ", best_score_perceptron)
    print("BEST PARAMS: ", best_params_perceptron)

    print("")
    print("")
    print("LOGISTIC REGRESSION RESULTS:")
    print("BEST SCORE: ", best_score_log)
    print("BEST PARAMS: ", best_params_log)


