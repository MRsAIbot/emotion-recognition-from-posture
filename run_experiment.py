import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import cross_validation

from pybrain.datasets import ClassificationDataSet
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

data_file1 = "actedData.csv"
data_file2 = "nonActedData.csv"
data_file3 = ""

def plot_confustion_matrix(cm):
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

	return 0

def load_data(location):
	data_frame = pd.read_csv(location)
	return data_frame

def ann_experiment(df, target_ind):
	classes = set(df.values[:,target_ind])
	target_dict = dict(itertools.izip(classes,range(len(classes))))

	ds = ClassificationDataSet(len(df.columns)-(target_ind+1), class_labels=classes)
	for i in range(len(df)):
		ds.appendLinked(df.values[i,target_ind+1::], target_dict[df.values[i,target_ind]])

	ds._convertToOneOfMany()
	net = buildNetwork(len(df.columns)-(target_ind+1), len(df.columns), len(classes), bias=True, hiddenclass=TanhLayer)

	trainer = BackpropTrainer(net, ds)
	print trainer.trainUntilConvergence(maxEpochs=10)
	print "Finished training Neural Network"
	return 0

def svm_experiment(df, target_ind, plot_cm=False):
	X = df.values[:,target_ind+1::].astype('float32')
	Y = df.values[:,target_ind]

	# Split the dataset in train and test sets
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
		X, Y, test_size=0.25, random_state=0)

	# Do grid search CV to find the best parameters for c and gamma
	c_range = 10.0 ** np.arange(-2,9)
	gamma_range = 10.0 ** np.arange(-5,4)
	param_grid = dict(gamma=gamma_range, C=c_range)
	cv = cross_validation.StratifiedKFold(y=Y_train, n_folds=3)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	grid.fit(X_train,Y_train)
	print "Test error after grid search CV: {0}".format(grid.score(X_test, Y_test))
	
	# Do a simple SVM
	clf_rbf = SVC()
	clf_rbf.fit(X_train,Y_train)
	print "Test error without proper parameter search: {0}".format(clf_rbf.score(X_test,Y_test))

	# Compute Confusion Matrix
	Y_pred = clf_rbf.predict(X_test)
	cm = confusion_matrix(Y_test,Y_pred)
	print(cm)

	if plot_cm:
		plot_confustion_matrix(cm)

	return 0

def main():
	df1 = load_data(data_file1)
	svm_experiment(df1,3,plot_cm=True)
	# ann_experiment(df1,3)

	# df2 = load_data(data_file2)
	# svm_experiment(df2,1)
	# ann_experiment(df2,1)

if __name__ == '__main__':
	main()