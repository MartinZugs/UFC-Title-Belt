import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# load pickled classifiers
lr_pickle = open("lr_clf.pickle", "rb")
lr_clf = pickle.load(lr_pickle)
lr_pickle.close()

svm_pickle = open("svm_clf.pickle", "rb")
svm_clf = pickle.load(svm_pickle)
svm_pickle.close()

per_pickle = open("per_clf.pickle", "rb")
per_clf = pickle.load(per_pickle)
per_pickle.close()

# print best models and scores for each classifier
print("____________________________________________________________________")
print("LR GRID SEARCH:")
print("____________________________________________________________________")
print("Best Estimator:")
print(lr_clf.best_estimator_)
print("Best Score:")
print(lr_clf.best_score_)
print("Best Params:")
print(lr_clf.best_params_)
print("Best Index:")
print(lr_clf.best_index_)
print()
#
# print("____________________________________________________________________")
# print("SVM GRID SEARCH:")
# print("____________________________________________________________________")
# print("Best Estimator:")
# print(svm_clf.best_estimator_)
# print("Best Score:")
# print(svm_clf.best_score_)
# print("Best Params:")
# print(svm_clf.best_params_)
# print("Best Index:")
# print(svm_clf.best_index_)
# print()
#
# print("____________________________________________________________________")
# print("PERCEPTRON GRID SEARCH:")
# print("____________________________________________________________________")
# print("Best Estimator:")
# print(per_clf.best_estimator_)
# print("Best Score:")
# print(per_clf.best_score_)
# print("Best Params:")
# print(per_clf.best_params_)
# print("Best Index:")
# print(per_clf.best_index_)
# print()

# LR grid search graphs
lr_grid_params = lr_clf.cv_results_['params']
lr_grid_scores = lr_clf.cv_results_['mean_test_score']

# # By C-Value
# lr_grid_c_values = []
# for i in range(len(lr_grid_params)):
#     lr_grid_c_values.append(lr_grid_params[i]['C'])
#     print('C: ' + str(lr_grid_c_values[i]) + ' - Score: ' + str(lr_grid_scores[i]))
#
# plt.axes(xscale='log')
# plt.scatter(lr_grid_c_values, lr_grid_scores, marker='o')
# plt.xlim(lr_grid_c_values[0]/10, lr_grid_c_values[-1] * 10)
# plt.ylim(lr_grid_scores[0] - .1, lr_grid_scores[-1] + .1)
# plt.xlabel('C-values')
# plt.ylabel('Mean Scores')
# plt.title('Logistic Regression Grid Search - Scores by C-value')
# plt.show()

# # By Class Weight
# lr_grid_class_weight_none = []
# lr_grid_class_weight_balanced = []
# for i in range(len(lr_grid_scores)):
#     if lr_grid_params[i]['class_weight'] == 'balanced':
#         lr_grid_class_weight_balanced.append(lr_grid_scores[i])
#     elif lr_grid_params[i]['class_weight'] == None:
#         lr_grid_class_weight_none.append(lr_grid_scores[i])
#     print('Class Weight: ' + str(lr_grid_params[i]['class_weight']) + ' - Score: ' + str(lr_grid_scores[i]))
# lr_class_weight_none_mean_score = sum(lr_grid_class_weight_none)/len(lr_grid_class_weight_none)
# lr_class_weight_balanced_mean_score = sum(lr_grid_class_weight_balanced)/len(lr_grid_class_weight_balanced)
#
# plt.bar([0, 1], [lr_class_weight_none_mean_score, lr_class_weight_balanced_mean_score], align='center')
# plt.xticks([0, 1], ['None', 'Balanced'])
# plt.ylim(.55, .7)
# plt.xlabel('Class Weight')
# plt.ylabel('Mean Scores')
# plt.title('Logistic Regression Grid Search - Average Score by Class Weight')
# plt.show()

# # By Class Weight
# lr_grid_warm_start_true = []
# lr_grid_warm_start_false = []
# for i in range(len(lr_grid_scores)):
#     if lr_grid_params[i]['warm_start'] == 'False':
#         lr_grid_warm_start_false.append(lr_grid_scores[i])
#     elif lr_grid_params[i]['warm_start'] == 'True':
#         lr_grid_warm_start_true.append(lr_grid_scores[i])
#     print('Warm Start: ' + str(lr_grid_params[i]['warm_start']) + ' - Score: ' + str(lr_grid_scores[i]))
# lr_warm_start_true_mean_score = sum(lr_grid_warm_start_true)/len(lr_grid_warm_start_true)
# lr_warm_start_false_mean_score = sum(lr_grid_warm_start_false)/len(lr_grid_warm_start_false)
#
# plt.bar([0, 1], [lr_warm_start_true_mean_score, lr_warm_start_false_mean_score], align='center')
# plt.xticks([0, 1], ['True', 'False'])
# plt.ylim(.55, .7)
# plt.xlabel('Warm Start')
# plt.ylabel('Mean Scores')
# plt.title('Logistic Regression Grid Search - Average Score by Warm Start')
# plt.show()

# for i in range(len(lr_grid_params)):
#     lr_grid_c_values.append(lr_grid_params[i]['C'])
#     print('C: ' + str(lr_grid_c_values[i]) + ' - Score: ' + str(lr_grid_scores[i]))

# get preprocessed, train/test split data
DATA_PATH = "manually-preprocessed_data-full-headers.csv"
pd_data = pd.read_csv(DATA_PATH)
pd_data_y = pd_data["Winner"]
pd_data_x = pd_data.loc[:, pd_data.columns != "Winner"]
x_train, x_test, y_train, y_test = train_test_split(pd_data_x, pd_data_y, test_size=0.2)
# train logistic regression with best classifier and test
lr_best = LogisticRegression(class_weight=None, penalty='l2', warm_start='True')
lr_trained = lr_best.fit(x_train, y_train)
print("Logistic Regression Best Model Score:" + str(lr_trained.score(x_test, y_test)))


# train svm with best classifier and test

# train perceptron with best classifier and test

