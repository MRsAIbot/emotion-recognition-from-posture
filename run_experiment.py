import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

data_file = "actedData.csv"

def load_data(location):
	data_frame = pd.read_csv(location)
	return data_frame

def main():
	df = load_data(data_file)
	print df

if __name__ == '__main__':
	main()