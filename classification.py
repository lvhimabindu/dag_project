import sys
import numpy as np
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold

# custom classes import
from my_exceptions import MyError

''' Helper class for learner. This class encapsulates all the necessary information to run a classification model from sklearn '''
''' Note: 
1. all the functions such as predict, predict_proba return np arrays and not lists
2. predict_proba returns a np array with a bunch of columns each of which corresponds to one class. This code returns probabilities of class 1 (assuming we are doing a binary classification with classes 0 and 1)
'''


class classification(object):

	# shared objects between all objects of the class (we dont change these anyway)
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes", "Logistic Regression"]
	classifiers = [ KNeighborsClassifier(10), SVC(kernel="linear", C=0.025, probability=True), SVC(gamma=2, C=1, probability=True), DecisionTreeClassifier(min_samples_leaf=10), RandomForestClassifier(min_samples_leaf=10, n_estimators=25), AdaBoostClassifier(), GaussianNB(), LogisticRegression(C=1e5)]

	# chosen model should be a string belonging to one of the "names" above. The constructor function creates the appropriate classifier object and assigns it to self.clf
	def __init__(self, chosen_model):

		# model_type should be a string
		if type(chosen_model).__name__ != 'str':
			raise MyError("Argument should be a string out of the following options: Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes, Logistic Regression")

		if chosen_model not in classification.names:
			raise MyError("Argument should be one of the following options: Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes, Logistic Regression")

		self.model_name = chosen_model
		self.ind = classification.names.index(chosen_model)
		self.clf = classification.classifiers[self.ind] # classifier object in the self.clf 

	# fit model on entire training data and return that model
	def fitEntireData(self, datapathorarr):

		if type(datapathorarr).__name__ == 'str':
			print ("Function fitEntireData: Assuming that the argument is filepath. Stop and re-run if this is incorrect!")
			dataset = self.readTrainingData(datapathorarr)
		else:
			print ("Function fitEntireData: Assuming that the argument is 2d np-array where last column has class labels. Stop and re-run if this is incorrect!")
			dataset = datapathorarr

		# reintializing the classifier to make sure that it does not remember any other training data
		self.clf = classification.classifiers[self.ind]

		colnum = len(dataset[0]) - 1

		X = dataset[:,0:colnum]
		Y = dataset[:,colnum]

		self.clf.fit(X, Y)

		return self.clf, self.clf.predict(X)


	# run the Model with the data set as an argument, dataset should be a 2d np array with last column being labels
	def runCVModel(self, datapathorarr):

		if type(datapathorarr).__name__ == 'str':
			print ("Function runCVModel: Assuming that the argument is filepath. Stop and re-run if this is incorrect!")
			dataset = self.readTrainingData(datapathorarr)
		else:
			print ("Function runCVModel: Assuming that the argument is 2d np-array where last column has class labels. Stop and re-run if this is incorrect!")
			dataset = datapathorarr

		# reinitializing the classifier to make sure that it does not have any memory of other training 
		self.clf = classification.classifiers[self.ind]
	
		colnum = len(dataset[0]) - 1

		X = dataset[:,0:colnum]
		Y = dataset[:,colnum]

		#scores = cross_validation.cross_val_predict(self.clf, Xvals, Yvals, cv=10)
		skf = StratifiedKFold(Y, n_folds=10, shuffle=True)

		indices = []
		pred_labels = []
		true_labels = []

		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]
			self.clf.fit(X_train, Y_train)
			pred_labels += list(self.clf.predict(X_test))
			true_labels += list(Y_test)
			indices += list(test_index)

		return pred_labels, true_labels, indices


	def getCVAUC(self, datapathorarr):

		# computes the AUC score given true values of Y and predicted Y scores. 
		y_preds, y_labels, indices = self.runCVModel(datapathorarr)
		correctness = [1 if y_preds[i]==y_labels[i] else 0 for i in indices]
		return (sum(correctness)+0.0)/len(indices)


	def readTrainingData(self, filepath):
		#print (type(np.loadtxt(filepath, dtype=int, delimiter=',')))
		return np.loadtxt(filepath, dtype=int, delimiter=',',skiprows=1)



