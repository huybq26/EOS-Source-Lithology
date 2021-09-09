import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier


from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense


import scipy.stats as stats
from keras.models import model_from_json
import numpy
import os


# load json and create model
json_file = open('model_mafic_pressure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_mafic_pressure.h5")
print("Loaded model from disk")
# evaluate loaded model on test data


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# read natural example


data2 = pd.read_excel('example.xlsx', header=0, index_col=0)


# train data determined from dataframe


Num_data = len(data2)
Naturedata = np.zeros((Num_data, 10))
for i in range(0, Num_data):
    for j in range(0, 10):
        Naturedata[i][j] = data2.iloc[i, j]


P = loaded_model.predict(Naturedata)
print(P)
