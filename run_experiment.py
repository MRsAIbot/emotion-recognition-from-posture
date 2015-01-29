import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import cross_validation

data_file = "actedData.csv"

def load_data(location):
	data_frame = pd.read_csv(location)
	return data_frame

def main():
	df = load_data(data_file)
	X = df.values[:,4::].astype('float32')
	Y = df.values[:,3]

	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
		X, Y, test_size=0.4, random_state=0)
	
	clf_rbf = SVC()
	clf_rbf.fit(X_train,Y_train)
	print clf_rbf.score(X_test,Y_test)

if __name__ == '__main__':
	main()