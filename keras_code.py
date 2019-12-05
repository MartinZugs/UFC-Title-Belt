# Binary Classification with Sonar Dataset: Baseline
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# path to the csv file
DATA_PATH = "manually-preprocessed_data-full-headers.csv" #'preprocessed_data_condensed.csv' # "manually-preprocessed_data-full-headers.csv"
PROCESSORS = 4
KFOLD = 10
VERBOSE = 1
RANDOM_STATE = 314159

# import data
pd_data = pd.read_csv(DATA_PATH)
pd_data = shuffle(pd_data, random_state=RANDOM_STATE)

# x and y data
pd_data_y = pd_data["Winner"]
pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]

pd_data_x_train, pd_data_x_test, pd_data_y_train, pd_data_y_test = train_test_split(pd_data_x, pd_data_y, test_size=.2, random_state=RANDOM_STATE)


# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=159, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


# evaluate model with standardized dataset
model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0) #verbose = 0 for nothing to show
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=PROCESSORS, cv=KFOLD, verbose=VERBOSE)
grid_result = grid.fit(pd_data_x_train, pd_data_y_train)


best_grid_result_score = accuracy_score(pd_data_y_test, grid_result.best_estimator_.predict(pd_data_x_test))

print()
print()
print("SVM BEST PARAMETERS : ")
print(grid_result.best_params_)
print()
print()
print("BEST SCORE ON THE TEST DATA IS : ", best_grid_result_score)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
