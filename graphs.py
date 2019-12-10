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

# # print best models and scores for each classifier
# print("____________________________________________________________________")
# print("LR GRID SEARCH:")
# print("____________________________________________________________________")
# print("Best Estimator:")
# print(lr_clf.best_estimator_)
# print("Best Score:")
# print(lr_clf.best_score_)
# print("Best Params:")
# print(lr_clf.best_params_)
# print("Best Index:")
# print(lr_clf.best_index_)
# print()
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

# # By Warm Start
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


# train svm with best classifier and test
# SVM grid search graphs
# svm_grid_params = svm_clf.cv_results_['params']
# svm_grid_scores = svm_clf.cv_results_['mean_test_score']

# # By C-Value
# svm_grid_c_values = []
# for i in range(len(svm_grid_params)):
#     svm_grid_c_values.append(svm_grid_params[i]['C'])
#     print('C: ' + str(svm_grid_c_values[i]) + ' - Score: ' + str(svm_grid_scores[i]))
#
# plt.axes(xscale='log')
# plt.scatter(svm_grid_c_values, svm_grid_scores, marker='o')
# plt.xlim(svm_grid_c_values[0]/10, svm_grid_c_values[-1] * 10)
# plt.ylim(svm_grid_scores[0] - .05, svm_grid_scores[-1] + .05)
# plt.xlabel('C-values')
# plt.ylabel('Mean Scores')
# plt.title('SVM Grid Search - Scores by C-value')
# plt.show()

# By gamma
# svm_grid_gamma = []
# for i in range(len(svm_grid_params)):
#     svm_grid_gamma.append(svm_grid_params[i]['C'])
#     print('Gamma: ' + str(svm_grid_gamma[i]) + ' - Score: ' + str(svm_grid_scores[i]))
# plt.axes(xscale='log')
# plt.scatter(svm_grid_gamma, svm_grid_scores, marker='o')
# plt.xlim(svm_grid_gamma[0]/10, svm_grid_gamma[-1] * 10)
# plt.ylim(svm_grid_scores[0] - .05, svm_grid_scores[-1] + .05)
# plt.xlabel('Gamma')
# plt.ylabel('Mean Scores')
# plt.title('SVM Grid Search - Scores by Gamma')
# plt.show()

# Perceptron grid search graphs
per_grid_params = per_clf.cv_results_['params']
per_grid_scores = per_clf.cv_results_['mean_test_score']
print()

# # By Warm Start
# per_grid_warm_start_true = []
# per_grid_warm_start_false = []
# for i in range(len(per_grid_scores)):
#     if per_grid_params[i]['warm_start'] == 'False':
#         per_grid_warm_start_false.append(per_grid_scores[i])
#     elif per_grid_params[i]['warm_start'] == 'True':
#         per_grid_warm_start_true.append(per_grid_scores[i])
#     print('Warm Start: ' + str(per_grid_params[i]['warm_start']) + ' - Score: ' + str(per_grid_scores[i]))
# per_warm_start_true_mean_score = sum(per_grid_warm_start_true)/len(per_grid_warm_start_true)
# per_warm_start_false_mean_score = sum(per_grid_warm_start_false)/len(per_grid_warm_start_false)
#
# plt.bar([0, 1], [per_warm_start_true_mean_score, per_warm_start_false_mean_score], align='center')
# plt.xticks([0, 1], ['True', 'False'])
# plt.ylim(.55, .7)
# plt.xlabel('Warm Start')
# plt.ylabel('Mean Scores')
# plt.title('Perceptron Grid Search - Average Score by Warm Start')
# plt.show()

# By Penalty
# per_grid_penalty_none = []
# per_grid_penalty_l1 = []
# per_grid_penalty_l2 = []
# per_grid_penalty_elasticnet = []
# for i in range(len(per_grid_scores)):
#     if per_grid_params[i]['penalty'] == 'l1':
#         per_grid_penalty_l1.append(per_grid_scores[i])
#     elif per_grid_params[i]['penalty'] == 'l2':
#         per_grid_penalty_l2.append(per_grid_scores[i])
#     elif per_grid_params[i]['penalty'] == 'None':
#         per_grid_penalty_none.append(per_grid_scores[i])
#     elif per_grid_params[i]['penalty'] == 'elasticnet':
#         per_grid_penalty_elasticnet.append(per_grid_scores[i])
#     print('Penalty: ' + str(per_grid_params[i]['penalty']) + ' - Score: ' + str(per_grid_scores[i]))
# per_penalty_none_mean_score = sum(per_grid_penalty_none)/len(per_grid_penalty_none)
# per_penalty_l1_mean_score = sum(per_grid_penalty_l1)/len(per_grid_penalty_l1)
# per_penalty_l2_mean_score = sum(per_grid_penalty_l2)/len(per_grid_penalty_l2)
# per_penalty_elasticnet_mean_score = sum(per_grid_penalty_elasticnet)/len(per_grid_penalty_elasticnet)
#
# plt.bar([0, 1, 2, 3], [per_penalty_none_mean_score, per_penalty_l1_mean_score, per_penalty_l2_mean_score, per_penalty_elasticnet_mean_score], align='center')
# plt.xticks([0, 1, 2, 3], ['None', 'l1', 'l2', 'elasticnet'])
# plt.ylim(.55, .7)
# plt.xlabel('Penalty')
# plt.ylabel('Mean Scores')
# plt.title('Perceptron Grid Search - Average Score by Penalty')
# plt.show()

# # By Class Weight
# per_grid_class_weight_balanced = []
# per_grid_class_weight_none = []
# for i in range(len(per_grid_scores)):
#     if per_grid_params[i]['class_weight'] == None:
#         per_grid_class_weight_none.append(per_grid_scores[i])
#     elif per_grid_params[i]['class_weight'] == 'balanced':
#         per_grid_class_weight_balanced.append(per_grid_scores[i])
#     print('Class Weight: ' + str(per_grid_params[i]['class_weight']) + ' - Score: ' + str(per_grid_scores[i]))
# per_class_weight_balanced_mean_score = sum(per_grid_class_weight_balanced)/len(per_grid_class_weight_balanced)
# per_class_weight_none_mean_score = sum(per_grid_class_weight_none)/len(per_grid_class_weight_none)
#
# plt.bar([0, 1], [per_class_weight_balanced_mean_score, per_class_weight_none_mean_score], align='center')
# plt.xticks([0, 1], ['Balanced', 'None'])
# plt.ylim(.55, .7)
# plt.xlabel('Class Weight')
# plt.ylabel('Mean Scores')
# plt.title('Perceptron Grid Search - Average Score by Class Weight')
# plt.show()
