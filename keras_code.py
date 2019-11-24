# Binary Classification with Sonar Dataset: Baseline
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# path to the csv file
DATA_PATH = "manually-preprocessed_data-full-headers.csv" #'preprocessed_data_condensed.csv' # "manually-preprocessed_data-full-headers.csv"
# import data
pd_data = pd.read_csv(DATA_PATH)
# x and y data
pd_data_y = pd_data["Winner"]
pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(2000, input_dim=159, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, pd_data_x, pd_data_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
