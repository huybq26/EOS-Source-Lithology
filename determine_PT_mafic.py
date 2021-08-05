#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:49:27 2021

@author: lilucheng
"""
# determine P/T ratio for mafic


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


data = pd.read_excel(
    'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1)

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


# melting degree
meltdegree = np.zeros((915, 1))
for i in range(0, 915):
    meltdegree[i] = data.iloc[i+1, 3]

# temperature
temperature = np.zeros((915, 1))
for i in range(0, 915):
    temperature[i] = data.iloc[i+1, 2]

# pressure
pressure = np.zeros((915, 1))
for i in range(0, 915):
    pressure[i] = data.iloc[i+1, 1]

# dry or not from dataframe
# 1 is hydrous  0 is anhydrous
Hydrous = np.zeros((915, 1))
for i in range(0, 915):
    Hydrous[i] = data.iloc[i+1, 29]


index1 = np.where((Group == 1) & (Hydrous == 1))  # hydrous
#index1 = np.where(Group == 1)

index_peridotite = index1[0]

index2 = np.where((Group == 2) & (Hydrous == 1))
index_transition = index2[0]

#index3 = np.where((Group == 3) & (Hydrous==1))
index3 = np.where(Group == 3)
index_mafic = index3[0]


# -------------------------------------------------------
# =============================================================================
# X = Traindata
# y = Group
#
#
#
#
# newX=X
# newy=y
#
# X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)
#
#
# clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
#
#
# clf = clf.fit(X_train, y_train)
#
# =============================================================================

# here we only consider mafic based on the data size
# mafic


meltdegree_mafic = meltdegree[index_mafic]

temperature_mafic = temperature[index_mafic]

pressure_mafic = pressure[index_mafic]


X_mafic = Traindata[index_mafic]  # traning data for mafic

hydrous_mafic = Hydrous[index_mafic]


# =============================================================================
# mafic

newX = X_mafic
# newy=md_label
# newy=tem_label
newy_md = meltdegree_mafic
newy_tem = temperature_mafic
newy_pre = pressure_mafic

newy_pt = 1000*newy_pre/newy_tem


# =============================================================================
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(10,)),
#     Dense(10, activation='relu'),
#     Dense(1, activation='sigmoid'),
# ])
#
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# =============================================================================


X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)


model = Sequential()

model.add(Dense(100, input_shape=(10,)))
model.add(Dense(100, activation='softsign'))  # 0.88
model.add(Dense(100, activation='softsign'))  # 0.88

# model.add(Dense(100, activation='tanh')) # 0.88


# tanh,exponential,linear
#model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='softplus'))
model.add(Dense(1, activation='linear'))


model.compile(optimizer='rmsprop',
              loss='mean_squared_error')


hist = model.fit(X_train, y_train,
                 batch_size=30, epochs=200,
                 validation_data=(X_test, y_test))


# ------------------------------------------------------------input the new sample
# here input the example


data2 = pd.read_excel('example.xlsx', header=0, index_col=0)

# train data determined from dataframe

Num_data = len(data2)
Naturedata = np.zeros((Num_data, 10))
for i in range(0, Num_data):
    for j in range(0, 10):
        Naturedata[i][j] = data2.iloc[i, j]


y_compare = model.predict(Naturedata)
y_compare = y_compare.flatten()


# plt.savefig('compare_lee_scatter.png',dpi=300)
# ------------------------version 2


plt.figure()
# histtype='step',

n, bins, patches = plt.hist(y_compare, 15, density=False,
                            facecolor='k', edgecolor='k', alpha=0.8, linewidth=2)


plt.xlabel('Predicted P/T (MPa/℃)', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.xlim(0.5, 4)
print("Mafic: ", y_compare)
# plt.savefig('compare_lee_histogram.png',dpi=300)
