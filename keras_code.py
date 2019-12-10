import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle

# path to the csv file
DATA_PATH = "manually-preprocessed_data-full-headers.csv"
# processors
PROCESSORS = -1
# K folds
KFOLD = 10
# verbosity
VERBOSE = 2
# random sttate
RANDOM_STATE = 314159

# import data
pd_data = pd.read_csv(DATA_PATH)
pd_data = shuffle(pd_data, random_state=RANDOM_STATE)

# x and y data
pd_data_y = pd_data["Winner"]
pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]
# separate training and test data
pd_data_x_train, pd_data_x_test, pd_data_y_train, pd_data_y_test = train_test_split(pd_data_x, pd_data_y, test_size=.2,
                                                                                    random_state=RANDOM_STATE)


# Create the KerasClassifier
def create_model(optimizer='rmsprop', kernel_init='glorot_uniform', activation='sigmoid', layer=1, node=32, dropout=.5):
    # create squential model
    model = Sequential()
    # add initial layer
    model.add(Dense(node, input_dim=DATA_SHAPE, kernel_initializer=kernel_init, activation=activation))
    model.add(Dropout(dropout))

    # add numbers of intermediate layers
    i = 1
    for _layer in range(layer):
        # print("ADDING LAYER : ", i)
        model.add(Dense(node, kernel_initializer=kernel_init, activation=activation))
        model.add(Dropout(dropout))
        i = i + 1

    # add the final layer
    model.add(Dense(1, kernel_initializer=kernel_init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# evaluate model with standardized dataset
model = KerasClassifier(build_fn=create_model, epochs=100, verbose=VERBOSE)  # verbose = 0 for nothing to show

# shape of the data. This defines all of the inputs
DATA_SHAPE = pd_data_x_train.shape[1]
DATA_LENGTH = pd_data_x_train.shape[0]


# grid search parameters
optimizers = ['rmsprop']
kernel_inits = ['uniform']
activations = ['sigmoid']
layers = [10]
nodes = [64]
dropouts = [0]
batches = [10]
epochs = [500]



# define parameter grid
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, kernel_init=kernel_inits,
                  activation=activations, layer=layers, node=nodes, dropout=dropouts)
# define gridsearch
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=PROCESSORS, cv=KFOLD, verbose=VERBOSE)
# fit
grid_result = grid.fit(pd_data_x_train, pd_data_y_train)

# save trained keras classifier to file
keras_outfile = open("keras3_clf.pickle", "w+b")
pickle.dump(grid_result.cv_results_, keras_outfile)
keras_outfile.close()

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_grid_result_score = accuracy_score(pd_data_y_test, grid_result.best_estimator_.predict(pd_data_x_test))

print()
print()
print("BEST PARAMETERS : ")
print(grid_result.best_params_)
print()
print()
print("BEST SCORE ON THE TEST DATA IS : ", best_grid_result_score)
