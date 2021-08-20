#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:10:25 2021

@author: lilucheng
"""
#mafic_web_P_and_T
#to determine P and T using mafic for web version



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
#1 is hydrous  0 is anhydrous
Hydrous = np.zeros((915, 1))
for i in range(0, 915):
    Hydrous[i] = data.iloc[i+1, 29]


index1 = np.where((Group == 1) & (Hydrous==1))  #hydrous
#index1 = np.where(Group == 1)

index_peridotite = index1[0]

index2 = np.where((Group == 2) & (Hydrous==1))
index_transition = index2[0]

#index3 = np.where((Group == 3) & (Hydrous==1))
index3 = np.where(Group == 3)
index_mafic = index3[0]


    


meltdegree_mafic = meltdegree[index_mafic]

temperature_mafic = temperature[index_mafic]

pressure_mafic = pressure[index_mafic]


X_mafic = Traindata[index_mafic]  # traning data for mafic

hydrous_mafic=Hydrous[index_mafic]


# =============================================================================
# mafic

newX = X_mafic
# newy=md_label
# newy=tem_label
newy_md=meltdegree_mafic
newy_tem = temperature_mafic
newy_pre=pressure_mafic

#newy_pt=1000*newy_pre/newy_tem

newy_pt=newy_tem/1000

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


#for p/t
#model.add(Dense(100, input_shape=(10,)))
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(1, activation='linear'))


#for temperature
model.add(Dense(100,activation='softsign') )

#model.add(Dense(100, activation='elu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88
model.add(Dense(100, activation='relu')) # 0.88


model.add(Dense(100, activation='relu')) # 0.88

model.add(Dense(1, activation='linear'))

#model.add(Dense(100, activation='tanh')) # 0.88


#tanh,exponential,linear

#model.add(Dense(1, activation='linear'))


model.compile(optimizer='rmsprop',
              loss='mean_squared_error')



hist = model.fit(X_train, y_train,
                 batch_size=20, epochs=300,
                 validation_data=(X_test, y_test))



y_pred=model.predict(newX)
y_pred=y_pred.flatten()
y_train=newy_pt.flatten()

rmse=math.sqrt(sum((y_pred-y_train)**2)/len(y_train))


#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#------------------------------------------------------------------------------------------
#pressure


newX = X_mafic
# newy=md_label
# newy=tem_label
newy_md=meltdegree_mafic
newy_tem = temperature_mafic
newy_pre=pressure_mafic

#newy_pt=1000*newy_pre/newy_tem

newy_pt=newy_pre



X_train, X_test, y_train, y_test = train_test_split(
    newX, newy_pt, train_size=0.8, random_state=0)



modelp = Sequential()


#for p/t
#model.add(Dense(100, input_shape=(10,)))
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(100, activation='softsign')) # 0.88
#model.add(Dense(1, activation='linear'))


#for temperature
modelp.add(Dense(100,activation='softsign') )

modelp.add(Dense(100, activation='relu')) # 0.88
modelp.add(Dense(100, activation='relu')) # 0.88
modelp.add(Dense(100, activation='relu')) # 0.88

modelp.add(Dense(1, activation='linear'))

#model.add(Dense(100, activation='tanh')) # 0.88


#tanh,exponential,linear

#model.add(Dense(1, activation='linear'))


modelp.compile(optimizer='rmsprop',
              loss='mean_squared_error')



histp = modelp.fit(X_train, y_train,
                 batch_size=20, epochs=200,
                 validation_data=(X_test, y_test))



yp_pred=modelp.predict(newX)
yp_pred=yp_pred.flatten()
yp_train=newy_pt.flatten()

rmsep=math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#read natural example



data2 = pd.read_excel('example.xlsx',header = 0,index_col=0)

#train data determined from dataframe

Num_data=len(data2)
Naturedata = np.zeros((Num_data,10))  
for i in range(0,Num_data):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i,j]  


T=model.predict(Naturedata)
P=modelp.predict(Naturedata)   
   


plt.figure()



#plot error bar 
plt.errorbar(T*1000,P, xerr=rmse*1000,yerr=rmsep, fmt='o', mfc='b',
         mec='k',ecolor='k',elinewidth=1,capthick=1,capsize=0)
ax = plt.gca()






plt.xlim(1000,1800)
plt.ylim(0,7)

ax.invert_yaxis()
plt.ylabel('Pressure (GPa)')
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部

#ax.set_xticklabels(row_labels, minor=False)

ax.set_xlabel('Temperature (℃)')    
ax.xaxis.set_label_position('top') 




