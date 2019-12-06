import pickle
from sklearn.linear_model import LogisticRegression

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

print("____________________________________________________________________")
print("SVM GRID SEARCH:")
print("____________________________________________________________________")
print("Best Estimator:")
print(svm_clf.best_estimator_)
print("Best Score:")
print(svm_clf.best_score_)
print("Best Params:")
print(svm_clf.best_params_)
print("Best Index:")
print(svm_clf.best_index_)
print()

print("____________________________________________________________________")
print("PERCEPTRON GRID SEARCH:")
print("____________________________________________________________________")
print("Best Estimator:")
print(per_clf.best_estimator_)
print("Best Score:")
print(per_clf.best_score_)
print("Best Params:")
print(per_clf.best_params_)
print("Best Index:")
print(per_clf.best_index_)
print()

# train logistic regression with best classifier

# lr = LogisticRegression(solver='lbfgs', random_state=, n_jobs=, max_iter=, verbose=)

