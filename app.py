from numpy.core.overrides import array_function_from_dispatcher
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import runpy
import openpyxl

import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error
from flask import send_file
from flask import send_file

from keras.models import Sequential
from keras.layers import Dense

import scipy.stats as stats
from io import BytesIO

app = Flask(__name__)
# app.config['UPLOAD_EXTENSIONS'] = ['.txt']

# y_compare

array_for_peridotite_temperature = []
array_for_mafic_temperature = []
array_for_transitional_temperature = []
array_for_peridotite_pressure = []
array_for_mafic_pressure = []
array_for_transitional_pressure = []


def run_first_model():
    data = pd.read_excel('data2_check.xlsx', header=None,
                         skipfooter=1, index_col=1, engine='openpyxl')

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

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

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

    return plt
    # plt.savefig("./static/image.png")


def run_transitional_model():
    data = pd.read_excel(
        'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1, engine='openpyxl')

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

    meltdegree_transition = meltdegree[index_transition]

    temperature_transition = temperature[index_transition]

    pressure_transition = pressure[index_transition]

    X_transition = Traindata[index_transition]  # traning data for mafic

    hydrous_transition = Hydrous[index_transition]

    # =============================================================================
    # mafic

    newX = X_transition
    # newy=md_label
    # newy=tem_label
    newy_md = meltdegree_transition
    newy_tem = temperature_transition
    newy_pre = pressure_transition

    # newy_pt=1000*newy_pre/newy_tem

    newy_pt = newy_tem/1000

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

    # for p/t
    #model.add(Dense(100, input_shape=(10,)))
    # model.add(Dense(100, activation='softsign')) # 0.88
    # model.add(Dense(100, activation='softsign')) # 0.88
    #model.add(Dense(1, activation='linear'))

    # for temperature
    model.add(Dense(100, activation='softsign'))

    # model.add(Dense(100, activation='elu')) # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(1, activation='linear'))

    # model.add(Dense(100, activation='tanh')) # 0.88

    # tanh,exponential,linear

    #model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')

    hist = model.fit(X_train, y_train,
                     batch_size=20, epochs=400,
                     validation_data=(X_test, y_test))

    y_pred = model.predict(newX)
    y_pred = y_pred.flatten()
    y_train = newy_pt.flatten()

    rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
    print('RMSE= %6.2f  ' % rmse)

    # =============================================================================
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # pressure

    newy_pt = newy_pre

    X_train, X_test, y_train, y_test = train_test_split(
        newX, newy_pt, train_size=0.8, random_state=0)

    modelp = Sequential()

    # for temperature
    modelp.add(Dense(100, activation='softsign'))
    modelp.add(Dense(100, activation='softsign'))

    modelp.add(Dense(1, activation='linear'))

    modelp.compile(optimizer='rmsprop',
                   loss='mean_squared_error')

    histp = modelp.fit(X_train, y_train,
                       batch_size=20, epochs=200,
                       validation_data=(X_test, y_test))

    yp_pred = modelp.predict(newX)
    yp_pred = yp_pred.flatten()
    yp_train = newy_pt.flatten()

    rmsep = math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))

    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # read natural example

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model.predict(Naturedata)
    P = modelp.predict(Naturedata)

    plt.figure()

    # plot error bar
    plt.errorbar(T*1000, P, xerr=rmse*1000, yerr=rmsep, fmt='o', mfc='b',
                 mec='k', ecolor='k', elinewidth=1, capthick=1, capsize=0)
    ax = plt.gca()

    plt.xlim(1000, 1800)
    plt.ylim(0, 7)

    ax.invert_yaxis()
    plt.ylabel('Pressure (GPa)')
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部

    #ax.set_xticklabels(row_labels, minor=False)

    ax.set_xlabel('Temperature (℃)')
    ax.xaxis.set_label_position('top')
    global array_for_transitional_pressure
    global array_for_transitional_temperature
    array_for_transitional_temperature = T
    array_for_transitional_pressure = P
    return plt

    # data = pd.read_excel(
    #     'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1)

    # # change into the data we need float
    # # train data determined from dataframe
    # Traindata = np.zeros((915, 10))
    # for i in range(0, 915):
    #     for j in range(0, 10):
    #         Traindata[i][j] = data.iloc[i+1, j+6]

    # # change nan into 0
    # for i in range(0, 915):
    #     for j in range(0, 10):
    #         if (np.isnan(Traindata[i][j])):
    #             Traindata[i][j] = 0

    # # lable from dataframe
    # Group = np.zeros((915, 1))
    # for i in range(0, 915):
    #     Group[i] = data.iloc[i+1, 24]

    # # melting degree
    # meltdegree = np.zeros((915, 1))
    # for i in range(0, 915):
    #     meltdegree[i] = data.iloc[i+1, 3]

    # # temperature
    # temperature = np.zeros((915, 1))
    # for i in range(0, 915):
    #     temperature[i] = data.iloc[i+1, 2]

    # # pressure
    # pressure = np.zeros((915, 1))
    # for i in range(0, 915):
    #     pressure[i] = data.iloc[i+1, 1]

    # # dry or not from dataframe
    # # 1 is hydrous  0 is anhydrous
    # Hydrous = np.zeros((915, 1))
    # for i in range(0, 915):
    #     Hydrous[i] = data.iloc[i+1, 29]

    # index1 = np.where((Group == 1) & (Hydrous == 1))  # hydrous
    # #index1 = np.where(Group == 1)

    # index_peridotite = index1[0]

    # index2 = np.where((Group == 2) & (Hydrous == 1))
    # index_transition = index2[0]

    # #index3 = np.where((Group == 3) & (Hydrous==1))
    # index3 = np.where(Group == 3)
    # index_mafic = index3[0]

    # # -------------------------------------------------------
    # # =============================================================================
    # # X = Traindata
    # # y = Group
    # #
    # #
    # #
    # #
    # # newX=X
    # # newy=y
    # #
    # # X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)
    # #
    # #
    # # clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    # #
    # #
    # # clf = clf.fit(X_train, y_train)
    # #
    # # =============================================================================

    # # here we only consider mafic based on the data size
    # # mafic
    # # this part will change based on the user select

    # meltdegree_mafic = meltdegree[index_transition]

    # temperature_mafic = temperature[index_transition]

    # pressure_mafic = pressure[index_transition]

    # X_mafic = Traindata[index_transition]  # traning data for mafic

    # hydrous_mafic = Hydrous[index_transition]

    # # =============================================================================
    # # mafic

    # newX = X_mafic
    # # newy=md_label
    # # newy=tem_label
    # newy_md = meltdegree_mafic
    # newy_tem = temperature_mafic
    # newy_pre = pressure_mafic

    # newy_pt = 1000*newy_pre/newy_tem

    # # =============================================================================
    # # model = Sequential([
    # #     Dense(10, activation='relu', input_shape=(10,)),
    # #     Dense(10, activation='relu'),
    # #     Dense(1, activation='sigmoid'),
    # # ])
    # #
    # #
    # # model.compile(loss='mean_squared_error', optimizer='adam')
    # # =============================================================================

    # X_train, X_test, y_train, y_test = train_test_split(
    #     newX, newy_pt, train_size=0.8, random_state=0)

    # model = Sequential()

    # model.add(Dense(100, input_shape=(10,)))
    # model.add(Dense(100, activation='softsign'))  # 0.88
    # model.add(Dense(100, activation='softsign'))  # 0.88

    # # model.add(Dense(100, activation='tanh')) # 0.88

    # # tanh,exponential,linear
    # #model.add(Dense(1, activation='linear'))
    # #model.add(Dense(1, activation='softplus'))
    # model.add(Dense(1, activation='linear'))

    # model.compile(optimizer='rmsprop',
    #               loss='mean_squared_error')

    # hist = model.fit(X_train, y_train,
    #                  batch_size=30, epochs=200,
    #                  validation_data=(X_test, y_test))

    # # ----------------------------------------------loss
    # plt.plot(hist.history['loss'])
    # plt.plot(hist.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper right')
    # # plt.show()

    # # --------------------------------------------accurancy

    # plt.figure()

    # y_pred = model.predict(newX)
    # y_pred = y_pred.flatten()

    # y_train = newy_pt.flatten()

    # r, s = stats.pearsonr(y_pred, y_train)

    # my = sum(y_train)/len(y_train)

    # r2_1 = 1-sum((y_pred-y_train)**2)/sum((y_train-my)**2)
    # print('${R_2}$= %6.2f  ' % r2_1)

    # rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
    # print('RMSE= %6.2f  ' % rmse)

    # # ---------------------------------------------------figure1  transtional

    # plt.scatter(y_train, y_pred, marker='o', facecolor='g', edgecolor='k')

    # #plt.legend(['Volatile-free and hydrous','Carbonated'], loc='upper left')
    # #name=('Overall data: ${R^2}$= %6.2f,    RMSE=%6.2f' %(r2_1, rmse))
    # #name=('Overall data: R= %6.2f,    RMSE=%6.2f MPa/℃' %(r, rmse))

    # # plt.title(name)
    # plt.text(1, 5, 'Volatile-free and hydrous\nT=850-1600℃\nP=0.5-6 GPa',
    #          fontsize=11, weight='bold')
    # #name_anhy=('${R^2}$= %6.2f\nRMSE=%6.2f' %(r2_anhy, rmse_anhy))
    # name = ('R= %6.2f\nRMSE=%6.2f MPa/℃' % (r, rmse))
    # plt.text(1, 4.1, name)

    # plt.title('Transitional')

    # # plt.text(1.2,0.2,name)
    # plt.ylabel('Predicted P/T (MPa/℃)', fontsize=12)
    # plt.xlabel('Experimental P/T (MPa/℃)', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # plt.plot([0, 11], [0, 11], 'k-')  # p/t
    # plt.plot([0, 11], [0.5, 11.5], 'k--')  # p/t
    # plt.plot([0, 11], [-0.5, 10.5], 'k--')  # p/t

    # plt.text(5.5, 6.7, '+0.5')
    # plt.text(6.6, 5.8, '-0.5')

    # plt.xlim(0, 7)
    # plt.ylim(0, 7)

    # # ------------------------------------------------------------------------------

    # # here input the example

    # data2 = pd.read_excel('./file/example.xlsx', header=0, index_col=0)

    # # train data determined from dataframe

    # Num_data = len(data2)
    # Naturedata = np.zeros((Num_data, 10))
    # for i in range(0, Num_data):
    #     for j in range(0, 10):
    #         Naturedata[i][j] = data2.iloc[i, j]

    # y_compare = model.predict(Naturedata)
    # y_compare = y_compare.flatten()

    # # plt.savefig('compare_lee_scatter.png',dpi=300)
    # # ------------------------version 2

    # plt.figure()
    # # histtype='step',

    # n, bins, patches = plt.hist(y_compare, 15, density=False,
    #                             facecolor='k', edgecolor='k', alpha=0.8, linewidth=2)

    # plt.xlabel('Predicted P/T (MPa/℃)', fontsize=12)
    # plt.ylabel('Number', fontsize=12)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    # plt.xlim(0.5, 4)
    # global array_for_transitional
    # array_for_transitional = y_compare
    # return plt
    # # plt.savefig('compare_lee_histogram.png',dpi=300)


def run_mafic_model():
    data = pd.read_excel('data2_check_dry.xlsx',
                         header=None, skipfooter=1, index_col=1, engine='openpyxl')
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

    # newy_pt=1000*newy_pre/newy_tem

    newy_pt = newy_tem/1000

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

    # for p/t
    #model.add(Dense(100, input_shape=(10,)))
    # model.add(Dense(100, activation='softsign')) # 0.88
    # model.add(Dense(100, activation='softsign')) # 0.88
    #model.add(Dense(1, activation='linear'))

    # for temperature
    model.add(Dense(100, activation='softsign'))

    # model.add(Dense(100, activation='elu')) # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(1, activation='linear'))

    # model.add(Dense(100, activation='tanh')) # 0.88

    # tanh,exponential,linear

    #model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')

    hist = model.fit(X_train, y_train,
                     batch_size=20, epochs=300,
                     validation_data=(X_test, y_test))

    y_pred = model.predict(newX)
    y_pred = y_pred.flatten()
    y_train = newy_pt.flatten()

    rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))

    # ==========================================================================================
    # ==========================================================================================
    # ==========================================================================================
    # ==========================================================================================
    # ------------------------------------------------------------------------------------------
    # pressure

    newX = X_mafic
    # newy=md_label
    # newy=tem_label
    newy_md = meltdegree_mafic
    newy_tem = temperature_mafic
    newy_pre = pressure_mafic

    # newy_pt=1000*newy_pre/newy_tem

    newy_pt = newy_pre

    X_train, X_test, y_train, y_test = train_test_split(
        newX, newy_pt, train_size=0.8, random_state=0)

    modelp = Sequential()

    # for p/t
    #model.add(Dense(100, input_shape=(10,)))
    # model.add(Dense(100, activation='softsign')) # 0.88
    # model.add(Dense(100, activation='softsign')) # 0.88
    #model.add(Dense(1, activation='linear'))

    # for temperature
    modelp.add(Dense(100, activation='softsign'))

    modelp.add(Dense(100, activation='relu'))  # 0.88
    modelp.add(Dense(100, activation='relu'))  # 0.88
    modelp.add(Dense(100, activation='relu'))  # 0.88

    modelp.add(Dense(1, activation='linear'))

    # model.add(Dense(100, activation='tanh')) # 0.88

    # tanh,exponential,linear

    #model.add(Dense(1, activation='linear'))

    modelp.compile(optimizer='rmsprop',
                   loss='mean_squared_error')

    histp = modelp.fit(X_train, y_train,
                       batch_size=20, epochs=200,
                       validation_data=(X_test, y_test))

    yp_pred = modelp.predict(newX)
    yp_pred = yp_pred.flatten()
    yp_train = newy_pt.flatten()

    rmsep = math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))

    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # read natural example

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model.predict(Naturedata)
    P = modelp.predict(Naturedata)

    plt.figure()

    # plot error bar
    plt.errorbar(T*1000, P, xerr=rmse*1000, yerr=rmsep, fmt='o', mfc='b',
                 mec='k', ecolor='k', elinewidth=1, capthick=1, capsize=0)
    ax = plt.gca()

    plt.xlim(1000, 1800)
    plt.ylim(0, 7)

    ax.invert_yaxis()
    plt.ylabel('Pressure (GPa)')
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部

    #ax.set_xticklabels(row_labels, minor=False)

    ax.set_xlabel('Temperature (℃)')
    ax.xaxis.set_label_position('top')

    global array_for_mafic_pressure
    global array_for_mafic_temperature
    array_for_mafic_temperature = T
    array_for_mafic_pressure = P
    return plt
#     data = pd.read_excel(
#         'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1)

#     # change into the data we need float
#     # train data determined from dataframe
#     Traindata = np.zeros((915, 10))
#     for i in range(0, 915):
#         for j in range(0, 10):
#             Traindata[i][j] = data.iloc[i+1, j+6]

#     # change nan into 0
#     for i in range(0, 915):
#         for j in range(0, 10):
#             if (np.isnan(Traindata[i][j])):
#                 Traindata[i][j] = 0

#     # lable from dataframe
#     Group = np.zeros((915, 1))
#     for i in range(0, 915):
#         Group[i] = data.iloc[i+1, 24]

#     # melting degree
#     meltdegree = np.zeros((915, 1))
#     for i in range(0, 915):
#         meltdegree[i] = data.iloc[i+1, 3]

#     # temperature
#     temperature = np.zeros((915, 1))
#     for i in range(0, 915):
#         temperature[i] = data.iloc[i+1, 2]

#     # pressure
#     pressure = np.zeros((915, 1))
#     for i in range(0, 915):
#         pressure[i] = data.iloc[i+1, 1]

#     # dry or not from dataframe
#     # 1 is hydrous  0 is anhydrous
#     Hydrous = np.zeros((915, 1))
#     for i in range(0, 915):
#         Hydrous[i] = data.iloc[i+1, 29]

#     index1 = np.where((Group == 1) & (Hydrous == 1))  # hydrous
#     #index1 = np.where(Group == 1)

#     index_peridotite = index1[0]

#     index2 = np.where((Group == 2) & (Hydrous == 1))
#     index_transition = index2[0]

#     #index3 = np.where((Group == 3) & (Hydrous==1))
#     index3 = np.where(Group == 3)
#     index_mafic = index3[0]

#     # -------------------------------------------------------
#     # =============================================================================
#     # X = Traindata
#     # y = Group
#     #
#     #
#     #
#     #
#     # newX=X
#     # newy=y
#     #
#     # X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)
#     #
#     #
#     # clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
#     #
#     #
#     # clf = clf.fit(X_train, y_train)
#     #
#     # =============================================================================

#     # here we only consider mafic based on the data size
#     # mafic

#     meltdegree_mafic = meltdegree[index_mafic]

#     temperature_mafic = temperature[index_mafic]

#     pressure_mafic = pressure[index_mafic]

#     X_mafic = Traindata[index_mafic]  # traning data for mafic

#     hydrous_mafic = Hydrous[index_mafic]

#     # =============================================================================
#     # mafic

#     newX = X_mafic
#     # newy=md_label
#     # newy=tem_label
#     newy_md = meltdegree_mafic
#     newy_tem = temperature_mafic
#     newy_pre = pressure_mafic

#     newy_pt = 1000*newy_pre/newy_tem

#     # =============================================================================
#     # model = Sequential([
#     #     Dense(10, activation='relu', input_shape=(10,)),
#     #     Dense(10, activation='relu'),
#     #     Dense(1, activation='sigmoid'),
#     # ])
#     #
#     #
#     # model.compile(loss='mean_squared_error', optimizer='adam')
#     # =============================================================================

#     X_train, X_test, y_train, y_test = train_test_split(
#         newX, newy_pt, train_size=0.8, random_state=0)

#     model = Sequential()

#     model.add(Dense(100, input_shape=(10,)))
#     model.add(Dense(100, activation='softsign'))  # 0.88
#     model.add(Dense(100, activation='softsign'))  # 0.88

#     # model.add(Dense(100, activation='tanh')) # 0.88

#     # tanh,exponential,linear
#     #model.add(Dense(1, activation='linear'))
#     #model.add(Dense(1, activation='softplus'))
#     model.add(Dense(1, activation='linear'))

#     model.compile(optimizer='rmsprop',
#                   loss='mean_squared_error')

#     hist = model.fit(X_train, y_train,
#                      batch_size=30, epochs=200,
#                      validation_data=(X_test, y_test))

#     # ------------------------------------------------------------input the new sample
#     # here input the example

#     data2 = pd.read_excel('./file/example.xlsx', header=0, index_col=0)

#     # train data determined from dataframe

#     Num_data = len(data2)
#     Naturedata = np.zeros((Num_data, 10))
#     for i in range(0, Num_data):
#         for j in range(0, 10):
#             Naturedata[i][j] = data2.iloc[i, j]

#     y_compare = model.predict(Naturedata)
#     y_compare = y_compare.flatten()

#     # plt.savefig('compare_lee_scatter.png',dpi=300)
#     # ------------------------version 2

#     plt.figure()
#     # histtype='step',

#     n, bins, patches = plt.hist(
#         y_compare, 15, density=False, facecolor='k', edgecolor='k', alpha=0.8, linewidth=2)

#     plt.xlabel('Predicted P/T (MPa/℃)', fontsize=12)
#     plt.ylabel('Number', fontsize=12)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     plt.xlim(0.5, 4)
#     global array_for_mafic
#     array_for_mafic = y_compare
#     # plt.savefig('compare_lee_histogram.png',dpi=300)
#     return plt


def run_peridotite_model():
    data = pd.read_excel(
        'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1, engine='openpyxl')

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

    meltdegree_peridotite = meltdegree[index_peridotite]

    temperature_peridotite = temperature[index_peridotite]

    pressure_peridotite = pressure[index_peridotite]

    X_peridotite = Traindata[index_peridotite]  # traning data for mafic

    # =============================================================================
    # peridotite

    newX = X_peridotite
    # newy=md_label2
    # newy=tem_label2
    newy_tem = temperature_peridotite
    newy_pre = pressure_peridotite

    newy_pt = newy_tem/1000

    X_train, X_test, y_train, y_test = train_test_split(
        newX, newy_pt, train_size=0.8, random_state=0)

    model = Sequential()

    # for temperature
    model.add(Dense(100, activation='softsign'))

    model.add(Dense(100, activation='elu'))  # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88
    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(100, activation='relu'))  # 0.88

    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')

    hist = model.fit(X_train, y_train,
                     batch_size=20, epochs=200,
                     validation_data=(X_test, y_test))

    # --------------------------------------------accurancy

    y_pred = model.predict(newX)
    y_pred = y_pred.flatten()
    y_train = newy_pt.flatten()

    rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))

    # ========================================================================================================
    # ========================================================================================================
    # peridotite  pressure

    newy_pt = newy_pre

    X_train, X_test, y_train, y_test = train_test_split(
        newX, newy_pt, train_size=0.8, random_state=0)

    modelp = Sequential()

    # for temperature
    modelp.add(Dense(100, activation='softsign'))
    modelp.add(Dense(100, activation='softsign'))

    # modelp.add(Dense(100, activation='relu')) # 0.88
    # modelp.add(Dense(100, activation='relu')) # 0.88

    # modelp.add(Dense(100, activation='relu')) # 0.88

    modelp.add(Dense(1, activation='linear'))

    modelp.compile(optimizer='rmsprop',
                   loss='mean_squared_error')

    histp = modelp.fit(X_train, y_train,
                       batch_size=20, epochs=200,
                       validation_data=(X_test, y_test))

    # --------------------------------------------accurancy

    yp_pred = modelp.predict(newX)
    yp_pred = yp_pred.flatten()

    yp_train = newy_pt.flatten()

    rmsep = math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))

    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # read natural example

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model.predict(Naturedata)
    P = modelp.predict(Naturedata)

    plt.figure()

    # plot error bar
    plt.errorbar(T*1000, P, xerr=rmse*1000, yerr=rmsep, fmt='o', mfc='b',
                 mec='k', ecolor='k', elinewidth=1, capthick=1, capsize=0)
    ax = plt.gca()

    plt.xlim(1000, 1800)
    plt.ylim(0, 7)

    ax.invert_yaxis()
    plt.ylabel('Pressure (GPa)')
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部

    #ax.set_xticklabels(row_labels, minor=False)

    ax.set_xlabel('Temperature (℃)')
    ax.xaxis.set_label_position('top')

    global array_for_peridotite_pressure
    global array_for_peridotite_temperature
    array_for_peridotite_temperature = T
    array_for_peridotite_pressure = P

    return plt

    # data = pd.read_excel(
    #     'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1)

    # # change into the data we need float
    # # train data determined from dataframe
    # Traindata = np.zeros((915, 10))
    # for i in range(0, 915):
    #     for j in range(0, 10):
    #         Traindata[i][j] = data.iloc[i+1, j+6]

    # # change nan into 0
    # for i in range(0, 915):
    #     for j in range(0, 10):
    #         if (np.isnan(Traindata[i][j])):
    #             Traindata[i][j] = 0

    # # lable from dataframe
    # Group = np.zeros((915, 1))
    # for i in range(0, 915):
    #     Group[i] = data.iloc[i+1, 24]

    # # melting degree
    # meltdegree = np.zeros((915, 1))
    # for i in range(0, 915):
    #     meltdegree[i] = data.iloc[i+1, 3]

    # # temperature
    # temperature = np.zeros((915, 1))
    # for i in range(0, 915):
    #     temperature[i] = data.iloc[i+1, 2]

    # # pressure
    # pressure = np.zeros((915, 1))
    # for i in range(0, 915):
    #     pressure[i] = data.iloc[i+1, 1]

    # # dry or not from dataframe
    # # 1 is hydrous  0 is anhydrous
    # Hydrous = np.zeros((915, 1))
    # for i in range(0, 915):
    #     Hydrous[i] = data.iloc[i+1, 29]

    # index1 = np.where((Group == 1) & (Hydrous == 1))  # hydrous
    # #index1 = np.where(Group == 1)

    # index_peridotite = index1[0]

    # index2 = np.where((Group == 2) & (Hydrous == 1))
    # index_transition = index2[0]

    # #index3 = np.where((Group == 3) & (Hydrous==1))
    # index3 = np.where(Group == 3)
    # index_mafic = index3[0]

    # # -------------------------------------------------------
    # # =============================================================================
    # # X = Traindata
    # # y = Group
    # #
    # #
    # #
    # #
    # # newX=X
    # # newy=y
    # #
    # # X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.9, random_state = 0)
    # #
    # #
    # # clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)
    # #
    # #
    # # clf = clf.fit(X_train, y_train)
    # #
    # # =============================================================================

    # # here we only consider mafic based on the data size
    # # mafic

    # meltdegree_peridotite = meltdegree[index_peridotite]

    # temperature_peridotite = temperature[index_peridotite]

    # pressure_peridotite = pressure[index_peridotite]

    # X_peridotite = Traindata[index_peridotite]  # traning data for mafic

    # # =============================================================================
    # # peridotite

    # newX = X_peridotite
    # # newy=md_label2
    # # newy=tem_label2
    # newy_tem = temperature_peridotite
    # newy_pre = pressure_peridotite

    # newy_pt = 1000*newy_pre/newy_tem

    # X_train, X_test, y_train, y_test = train_test_split(
    #     newX, newy_pt, train_size=0.8, random_state=0)

    # model = Sequential()

    # model.add(Dense(30, input_shape=(10,)))
    # model.add(Dense(30, activation='softsign'))

    # model.add(Dense(1, activation='linear'))
    # #model.add(Dense(1, activation='softplus'))

    # model.compile(optimizer='rmsprop',
    #               loss='mean_squared_error')

    # hist = model.fit(X_train, y_train,
    #                  batch_size=30, epochs=100,
    #                  validation_data=(X_test, y_test))

    # # ------------------------------------------------------------input the new sample
    # # here input the example

    # data2 = pd.read_excel('./file/example.xlsx', header=0, index_col=0)

    # # train data determined from dataframe

    # Num_data = len(data2)
    # Naturedata = np.zeros((Num_data, 10))
    # for i in range(0, Num_data):
    #     for j in range(0, 10):
    #         Naturedata[i][j] = data2.iloc[i, j]

    # y_compare = model.predict(Naturedata)
    # y_compare = y_compare.flatten()

    # # plt.savefig('compare_lee_scatter.png',dpi=300)
    # # ------------------------version 2

    # plt.figure()
    # # histtype='step',

    # n, bins, patches = plt.hist(
    #     y_compare, 15, density=False, facecolor='k', edgecolor='k', alpha=0.8, linewidth=2)

    # plt.xlabel('Predicted P/T (MPa/℃)', fontsize=12)
    # plt.ylabel('Number', fontsize=12)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    # plt.xlim(0.5, 4)
    # global array_for_peridotite
    # array_for_peridotite = y_compare
    # return plt
    # # plt.savefig('compare_lee_histogram.png',dpi=300)


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
        str1 += " "
    return str1


def run_modify_excel():
    global array_for_peridotite_temperature
    global array_for_mafic_temperature
    global array_for_transitional_temperature
    global array_for_peridotite_pressure
    global array_for_mafic_pressure
    global array_for_transitional_pressure
    array_for_peridotite_temperature = np.concatenate(
        array_for_peridotite_temperature, axis=0)
    array_for_mafic_temperature = np.concatenate(
        array_for_mafic_temperature, axis=0)
    array_for_transitional_temperature = np.concatenate(
        array_for_transitional_temperature, axis=0)
    array_for_peridotite_pressure = np.concatenate(
        array_for_peridotite_pressure, axis=0)
    array_for_mafic_pressure = np.concatenate(array_for_mafic_pressure, axis=0)
    array_for_transitional_pressure = np.concatenate(
        array_for_transitional_pressure, axis=0)

    theFile = openpyxl.load_workbook('./file/example.xlsx')
    arr = theFile.sheetnames
    # print(arr[0])
    currentSheet = theFile[arr[0]]
    print(currentSheet['B4'].value)
    currentSheet['L1'] = "Peridotite Pressure"
    currentSheet['M1'] = "Peridotite Temperature"
    currentSheet['N1'] = "Mafic Pressure"
    currentSheet['O1'] = "Mafic Temperature"
    currentSheet['P1'] = "Transitional Pressure"
    currentSheet['Q1'] = "Transitional Temperature"
    # currentSheet['M2'] = listToString(sample_array)
    marker_row = ""
    for i in range(0, len(array_for_peridotite_pressure)):
        current_column = "L"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_peridotite_pressure[i]
        marker_row = current_row

    for i in range(0, len(array_for_peridotite_temperature)):
        current_column = "M"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_peridotite_temperature[i]
        marker_row = current_row

    for i in range(0, len(array_for_mafic_pressure)):
        current_column = "N"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_mafic_pressure[i]
        marker_row = current_row

    for i in range(0, len(array_for_mafic_temperature)):
        current_column = "O"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_mafic_temperature[i]
        marker_row = current_row

    for i in range(0, len(array_for_transitional_pressure)):
        current_column = "P"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_transitional_pressure[i]
        marker_row = current_row

    for i in range(0, len(array_for_transitional_temperature)):
        current_column = "Q"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = array_for_transitional_temperature[i]
        marker_row = current_row

    marker_location = "L"+str(int(marker_row)+1)
    theFile.save("./static/result.xlsx")
    return 'a string'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    if os.path.isfile('./file/example.xlsx'):
        os.remove("./file/example.xlsx")
    if os.path.isfile('./static/result.xlsx'):
        os.remove("./static/result.xlsx")
    if request.method == 'POST':
        input_file = request.files["upload-file"]
        if input_file.filename != '':
            input_file.save("./file/example.xlsx")
            runpy.run_path(path_name='ANN_classification_web.py')
    # print(array_for_peridotite)
    return render_template("index.html")


@app.route('/fig')
def fig():
    if os.path.isfile('./file/example.xlsx'):
        fig = run_first_model()
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        # os.remove("./file/example.xlsx")  # remove Excel from file system after use
        return send_file(img, mimetype='image/png')
    return 'a string'


@app.route('/transitional')
def transitional():
    if os.path.isfile('./file/example.xlsx'):
        fig = run_transitional_model()
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        # os.remove("./file/example.xlsx")  # remove Excel from file system after use
        return send_file(img, mimetype='image/png')
    return 'a string'


@app.route('/mafic')
def mafic():
    if os.path.isfile('./file/example.xlsx'):
        fig = run_mafic_model()
        run_modify_excel()
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        # os.remove("./file/example.xlsx")  # remove Excel from file system after use
        return send_file(img, mimetype='image/png')
    return 'a string'


@app.route('/peridotite')
def peridotite():
    if os.path.isfile('./file/example.xlsx'):
        fig = run_peridotite_model()
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        # os.remove("./file/example.xlsx")  # remove Excel from file system after use
        return send_file(img, mimetype='image/png')
    return 'a string'


# @app.errorhandler(404)
# def not_found_error(error):
#     return render_template('404.html'), 404

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


# @app.errorhandler(500)
# def internal(e):
#     return jsonify(error=str(e)), 500

# Handling error 500 and displaying relevant web page


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html")


if __name__ == "__main__":
    app.run(debug=True)
