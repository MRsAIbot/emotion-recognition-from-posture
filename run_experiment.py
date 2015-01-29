import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import cross_validation

from pybrain.datasets import ClassificationDataSet
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

data_file = "actedData.csv"

def load_data(location):
	data_frame = pd.read_csv(location)
	return data_frame

def ann_experiment(df):
	classes = set(df.values[:,3])
	target_dict = dict(itertools.izip(classes,range(len(classes))))

	ds = ClassificationDataSet(len(df.columns)-4, class_labels=classes)
	for i in range(len(df)):
		ds.appendLinked(df.values[i,4::], target_dict[df.values[i,3]])

	ds._convertToOneOfMany()
	net = buildNetwork(len(df.columns)-4, len(df.columns), len(classes), bias=True, hiddenclass=TanhLayer)

	trainer = BackpropTrainer(net, ds)
	print trainer.train()
	print "Finished training Neural Network"
	return 0

def svm_experiment(df):
	X = df.values[:,4::].astype('float32')
	Y = df.values[:,3]

	# Split the dataset in train and test sets
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
		X, Y, test_size=0.4, random_state=0)

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

	return 0

def main():
	df = load_data(data_file)
	svm_experiment(df)
	ann_experiment(df)

if __name__ == '__main__':
	main()