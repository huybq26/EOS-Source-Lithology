#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:18:41 2021

@author: lilucheng
"""
# this classification model for web used
# inputs: data from the Excel files
# outputs: three labels(Peridotite, transitional, mafic)


import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error


# -------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2_check.xlsx', header=None,
                     skipfooter=1, index_col=1)

# change into the data we need float
# train data determined from dataframe
Traindata = np.zeros((915, 10))
for i in range(0, 915):
    for j in range(0, 10):
        Traindata[i][j] = data.iloc[i+1, j+6]


# change nan into 0
for i in range(0, 915):
    for j in range(0, 10):
        if (np.isnan(Traindata[i][j])):
            Traindata[i][j] = 0


# lable from dataframe
Group = np.zeros((915, 1))
for i in range(0, 915):
    Group[i] = data.iloc[i+1, 24]


# -------------------------------------------------------
X = Traindata
y = Group


# D=X
#idq=np.where((D[:,0]<30) | (D[:,0]>65))
# idq0=idq[0]
# newX=np.delete(D,idq0,0)

# newy=np.delete(y,idq0)
# newy=newy.reshape(-1,1)

newX = X
newy = y

X_train, X_test, y_train, y_test = train_test_split(
    newX, newy, train_size=0.8, random_state=0)


clf = MLPClassifier(activation='relu', solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1)


clf = clf.fit(X_train, y_train)


accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network test:', accuracy_ANN)


# visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)


y_result = clf.predict(newX)
y_result = y_result.reshape(-1, 1)
difference_total = np.zeros((915, 1))
difference_total = newy-y_result
idx = np.where(difference_total == 0)
idx0 = idx[0]
Same_alldata = len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)


# ------------------------------------------------------------------------------------------
# read data from example

data2 = pd.read_excel('example.xlsx', header=0, index_col=0)

# train data determined from dataframe

Num_data = len(data2)
Naturedata = np.zeros((Num_data, 10))
for i in range(0, Num_data):
    for j in range(0, 10):
        Naturedata[i][j] = data2.iloc[i, j]

Naturedata_result = clf.predict(Naturedata)

Naturedata_result = Naturedata_result.reshape(-1, 1)


# chose natural_results by ANN== 1,2,3


idxd1 = np.where((Naturedata_result == 1))
idxd10 = idxd1[0]

idxd2 = np.where((Naturedata_result == 2))  # transitional
idxd20 = idxd2[0]

#idxd3=(np.where(difference_total!=0) and np.where(Naturedata_result==3))

idxd3 = np.where((Naturedata_result == 3))  # mafic
idxd30 = idxd3[0]


#########


# show the results by figures
MgO = np.zeros((Num_data, 1))
for i in range(0, Num_data):
    MgO[i] = data2.iloc[i, 6]


CaO = np.zeros((Num_data, 1))
for i in range(0, Num_data):
    CaO[i] = data2.iloc[i, 7]


###############################
plt.figure()


l1 = plt.scatter(MgO[idxd30], CaO[idxd30], marker='^',
                 c='red', edgecolors='0.5', s=30, linewidth=1)
# plt.scatter(Mg[idxd0],fcmsd[idxd0],marker='+',c='r',edgecolors='b',s=15,linewidth=0.5)

l2 = plt.scatter(MgO[idxd20], CaO[idxd20], marker='s',
                 c='limegreen', edgecolors='k', s=30, linewidth=1)

l3 = plt.scatter(MgO[idxd10], CaO[idxd10], marker='x',
                 c='deepskyblue', edgecolors='deepskyblue', s=30, linewidth=1)


plt.legend([l1, l2, l3], ['Mafic', 'Transitional', 'Peridotite'],
           loc='lower left', fontsize=10)
plt.xlabel('MgO(wt%)', fontsize=12)

plt.ylabel('CaO(wt%)', fontsize=12)

plt.savefig("./static/image.png")
