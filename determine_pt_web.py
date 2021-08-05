#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:55:11 2021

@author: lilucheng
"""
# determine p/t ratio web
# using example.xlsx


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
# this part will change based on the user select


meltdegree_mafic = meltdegree[index_transition]

temperature_mafic = temperature[index_transition]

pressure_mafic = pressure[index_transition]


X_mafic = Traindata[index_transition]  # traning data for mafic

hydrous_mafic = Hydrous[index_transition]


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


# ----------------------------------------------loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# --------------------------------------------accurancy

plt.figure()

y_pred = model.predict(newX)
y_pred = y_pred.flatten()

y_train = newy_pt.flatten()

r, s = stats.pearsonr(y_pred, y_train)

my = sum(y_train)/len(y_train)

r2_1 = 1-sum((y_pred-y_train)**2)/sum((y_train-my)**2)
print('${R_2}$= %6.2f  ' % r2_1)


rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
print('RMSE= %6.2f  ' % rmse)

# ---------------------------------------------------figure1  transtional

plt.scatter(y_train, y_pred, marker='o', facecolor='g', edgecolor='k')

#plt.legend(['Volatile-free and hydrous','Carbonated'], loc='upper left')
#name=('Overall data: ${R^2}$= %6.2f,    RMSE=%6.2f' %(r2_1, rmse))
#name=('Overall data: R= %6.2f,    RMSE=%6.2f MPa/℃' %(r, rmse))

# plt.title(name)
plt.text(1, 5, 'Volatile-free and hydrous\nT=850-1600℃\nP=0.5-6 GPa',
         fontsize=11, weight='bold')
#name_anhy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_anhy, rmse_anhy))
name = ('R= %6.2f\nRMSE=%6.2f MPa/℃' % (r, rmse))
plt.text(1, 4.1, name)


plt.title('Transitional')


# plt.text(1.2,0.2,name)
plt.ylabel('Predicted P/T (MPa/℃)', fontsize=12)
plt.xlabel('Experimental P/T (MPa/℃)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.plot([0, 11], [0, 11], 'k-')  # p/t
plt.plot([0, 11], [0.5, 11.5], 'k--')  # p/t
plt.plot([0, 11], [-0.5, 10.5], 'k--')  # p/t


plt.text(5.5, 6.7, '+0.5')
plt.text(6.6, 5.8, '-0.5')


plt.xlim(0, 7)
plt.ylim(0, 7)


# ------------------------------------------------------------------------------

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
print("Transitional: ", y_compare)
# plt.savefig('compare_lee_histogram.png',dpi=300)
