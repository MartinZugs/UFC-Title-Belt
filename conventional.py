import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import sys

class Result:
    def __init__(self):
        pass


def run_k_fold_batch(in_data_x, in_data_y, in_k_fold, in_random_state):

    # k fold definition
    kf = KFold(n_splits=K_FOLDS,shuffle=True, random_state=RANDOM_STATE)
    kf.get_n_splits()

    for train_index, test_index in kf.split(pd_data_x):

        print()
        # print("TRAIN:", train_index, "TEST:", test_index)

        x_train, x_test = pd_data_x.iloc[train_index], pd_data_x.iloc[test_index]
        y_train, y_test = pd_data_y.iloc[train_index], pd_data_y.iloc[test_index]

        logreg = LogisticRegression(solver='lbfgs')
        logreg.fit(x_train, y_train)
        score = logreg.score(x_test, y_test)
        print( score )


if __name__ == '__main__':

    # path to the csv file
    DATA_PATH = "manually-preprocessed_data-full-headers.csv"
    # number of folds
    K_FOLDS = 10
    # random state to use for shuffling
    RANDOM_STATE = 987654321

    # import data
    pd_data = pd.read_csv(DATA_PATH)

    # x and y data
    pd_data_y = pd_data["Winner"]
    pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]

    run_k_fold_batch(in_data_x=pd_data_x, in_data_y=pd_data_y, in_k_fold=K_FOLDS, in_random_state=RANDOM_STATE)


