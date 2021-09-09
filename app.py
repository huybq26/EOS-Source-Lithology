from numpy.core.overrides import array_function_from_dispatcher
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import runpy
import openpyxl

from keras.models import model_from_json
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
nature_data = []
rmse_mafic = 0
rmse_transitional = 0
rmse_peridotite = 0
rmsep_mafic = 0
rmsep_transitional = 0
rmsep_peridotite = 0


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
    global nature_data
    nature_data = Naturedata_result
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

    plt.savefig("./static/first.png")

    return plt
    # plt.savefig("./static/image.png")


def run_transitional_model():
    # data = pd.read_excel(
    #     'data2_check_dry.xlsx', header=None, skipfooter=1, index_col=1, engine='openpyxl')

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

    # meltdegree_transition = meltdegree[index_transition]

    # temperature_transition = temperature[index_transition]

    # pressure_transition = pressure[index_transition]

    # X_transition = Traindata[index_transition]  # traning data for mafic

    # hydrous_transition = Hydrous[index_transition]

    # # =============================================================================
    # # mafic

    # newX = X_transition
    # # newy=md_label
    # # newy=tem_label
    # newy_md = meltdegree_transition
    # newy_tem = temperature_transition
    # newy_pre = pressure_transition

    # # newy_pt=1000*newy_pre/newy_tem

    # newy_pt = newy_tem/1000

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

    # # for p/t
    # #model.add(Dense(100, input_shape=(10,)))
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # #model.add(Dense(1, activation='linear'))

    # # for temperature
    # model.add(Dense(100, activation='softsign'))

    # # model.add(Dense(100, activation='elu')) # 0.88
    # model.add(Dense(100, activation='relu'))  # 0.88
    # model.add(Dense(100, activation='relu'))  # 0.88

    # model.add(Dense(100, activation='relu'))  # 0.88

    # model.add(Dense(1, activation='linear'))

    # # model.add(Dense(100, activation='tanh')) # 0.88

    # # tanh,exponential,linear

    # #model.add(Dense(1, activation='linear'))

    # model.compile(optimizer='rmsprop',
    #               loss='mean_squared_error')

    # hist = model.fit(X_train, y_train,
    #                  batch_size=20, epochs=400,
    #                  validation_data=(X_test, y_test))

    # y_pred = model.predict(newX)
    # y_pred = y_pred.flatten()
    # y_train = newy_pt.flatten()

    # rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))
    # global rmse_transitional
    # rmse_transitional = rmse
    # print('RMSE= %6.2f  ' % rmse)

    # # =============================================================================
    # # ------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------
    # # pressure

    # newy_pt = newy_pre

    # X_train, X_test, y_train, y_test = train_test_split(
    #     newX, newy_pt, train_size=0.8, random_state=0)

    # modelp = Sequential()

    # # for temperature
    # modelp.add(Dense(100, activation='softsign'))
    # modelp.add(Dense(100, activation='softsign'))

    # modelp.add(Dense(1, activation='linear'))

    # modelp.compile(optimizer='rmsprop',
    #                loss='mean_squared_error')

    # histp = modelp.fit(X_train, y_train,
    #                    batch_size=20, epochs=200,
    #                    validation_data=(X_test, y_test))

    # yp_pred = modelp.predict(newX)
    # yp_pred = yp_pred.flatten()
    # yp_train = newy_pt.flatten()

    # rmsep = math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))

    # # --------------------------------------------------------------------------------------
    # # --------------------------------------------------------------------------------------
    # # read natural example
    rmsep = 0.3508304773262083
    rmse = 0.08881064833319496
    json_file = open('model_temperature_transitional.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_temperature_transitional.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    json_file_2 = open('model_pressure_transitional.json', 'r')
    loaded_model_json_2 = json_file_2.read()
    json_file_2.close()
    modelp = model_from_json(loaded_model_json_2)
    # load weights into new model
    modelp.load_weights("model_pressure_transitional.h5")
    print("Loaded model from disk")

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model(Naturedata)
    P = modelp(Naturedata)

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
    global rmsep_transitional
    rmsep_transitional = rmsep
    global rmse_transitional
    rmse_transitional = rmse
    plt.savefig('./static/transitional.png', dpi=300)
    return plt
    # return 'a string'


def run_mafic_model():
    # data = pd.read_excel('data2_check_dry.xlsx',
    #                      header=None, skipfooter=1, index_col=1, engine='openpyxl')
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

    # meltdegree_mafic = meltdegree[index_mafic]

    # temperature_mafic = temperature[index_mafic]

    # pressure_mafic = pressure[index_mafic]

    # X_mafic = Traindata[index_mafic]  # traning data for mafic

    # hydrous_mafic = Hydrous[index_mafic]

    # # =============================================================================
    # # mafic

    # newX = X_mafic
    # # newy=md_label
    # # newy=tem_label
    # newy_md = meltdegree_mafic
    # newy_tem = temperature_mafic
    # newy_pre = pressure_mafic

    # # newy_pt=1000*newy_pre/newy_tem

    # newy_pt = newy_tem/1000

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

    # # for p/t
    # #model.add(Dense(100, input_shape=(10,)))
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # #model.add(Dense(1, activation='linear'))

    # # for temperature
    # model.add(Dense(100, activation='softsign'))

    # # model.add(Dense(100, activation='elu')) # 0.88
    # model.add(Dense(100, activation='relu'))  # 0.88
    # model.add(Dense(100, activation='relu'))  # 0.88

    # model.add(Dense(100, activation='relu'))  # 0.88

    # model.add(Dense(1, activation='linear'))

    # # model.add(Dense(100, activation='tanh')) # 0.88

    # # tanh,exponential,linear

    # #model.add(Dense(1, activation='linear'))

    # model.compile(optimizer='rmsprop',
    #               loss='mean_squared_error')

    # hist = model.fit(X_train, y_train,
    #                  batch_size=20, epochs=300,
    #                  validation_data=(X_test, y_test))

    # y_pred = model.predict(newX)
    # y_pred = y_pred.flatten()
    # y_train = newy_pt.flatten()

    # rmse = math.sqrt(sum((y_pred-y_train)**2)/len(y_train))

    # # ==========================================================================================
    # # ==========================================================================================
    # # ==========================================================================================
    # # ==========================================================================================
    # # ------------------------------------------------------------------------------------------
    # # pressure

    # newX = X_mafic
    # # newy=md_label
    # # newy=tem_label
    # newy_md = meltdegree_mafic
    # newy_tem = temperature_mafic
    # newy_pre = pressure_mafic

    # # newy_pt=1000*newy_pre/newy_tem

    # newy_pt = newy_pre

    # X_train, X_test, y_train, y_test = train_test_split(
    #     newX, newy_pt, train_size=0.8, random_state=0)

    # modelp = Sequential()

    # # for p/t
    # #model.add(Dense(100, input_shape=(10,)))
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # # model.add(Dense(100, activation='softsign')) # 0.88
    # #model.add(Dense(1, activation='linear'))

    # # for temperature
    # modelp.add(Dense(100, activation='softsign'))

    # modelp.add(Dense(100, activation='relu'))  # 0.88
    # modelp.add(Dense(100, activation='relu'))  # 0.88
    # modelp.add(Dense(100, activation='relu'))  # 0.88

    # modelp.add(Dense(1, activation='linear'))

    # # model.add(Dense(100, activation='tanh')) # 0.88

    # # tanh,exponential,linear

    # #model.add(Dense(1, activation='linear'))

    # modelp.compile(optimizer='rmsprop',
    #                loss='mean_squared_error')

    # histp = modelp.fit(X_train, y_train,
    #                    batch_size=20, epochs=200,
    #                    validation_data=(X_test, y_test))

    # yp_pred = modelp.predict(newX)
    # yp_pred = yp_pred.flatten()
    # yp_train = newy_pt.flatten()

    # rmsep = math.sqrt(sum((yp_pred-yp_train)**2)/len(yp_train))

    # # --------------------------------------------------------------------------------------
    # # --------------------------------------------------------------------------------------
    # # read natural example
    # load json and create model
    rmsep = 0.360724512165989
    rmse = 0.050046018220376624
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    json_file_2 = open('model_mafic_pressure.json', 'r')
    loaded_model_json_2 = json_file_2.read()
    json_file_2.close()
    modelp = model_from_json(loaded_model_json_2)
    # load weights into new model
    modelp.load_weights("model_mafic_pressure.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model(Naturedata)
    P = modelp(Naturedata)

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
    global rmse_mafic
    rmse_mafic = rmse
    global rmsep_mafic
    rmsep_mafic = rmsep
    plt.savefig('./static/mafic.png', dpi=300)
    return plt


def run_peridotite_model():
    rmsep = 0.2457602527225323
    rmse = 0.05728688422144199

    json_file = open('model_temperature_peridotitic.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_temperature_peridotitic.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    json_file_2 = open('model_pressure_peridotitic.json', 'r')
    loaded_model_json_2 = json_file_2.read()
    json_file_2.close()
    modelp = model_from_json(loaded_model_json_2)
    # load weights into new model
    modelp.load_weights("model_pressure_peridotitic.h5")
    print("Loaded model from disk")

    data2 = pd.read_excel('./file/example.xlsx', header=0,
                          index_col=0, engine='openpyxl')

    # train data determined from dataframe

    Num_data = len(data2)
    Naturedata = np.zeros((Num_data, 10))
    for i in range(0, Num_data):
        for j in range(0, 10):
            Naturedata[i][j] = data2.iloc[i, j]

    T = model(Naturedata)
    P = modelp(Naturedata)

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
    global rmse_peridotite
    rmse_peridotite = rmse
    global rmsep_peridotite
    rmsep_peridotite = rmsep
    plt.savefig('./static/peridotite.png', dpi=300)
    return plt


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
    global nature_data
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
    nature_data = np.concatenate(nature_data, axis=0)

    theFile = openpyxl.load_workbook('./file/example.xlsx')
    arr = theFile.sheetnames
    # print(arr[0])
    currentSheet = theFile[arr[0]]
    print(currentSheet['B4'].value)
    currentSheet['L1'] = "Peridotite Pressure (GPa)"
    currentSheet['M1'] = "Peridotite Pressure Error"
    currentSheet['M2'] = float(
        "{0:.2f}".format(round(rmsep_peridotite, 2)))
    currentSheet['N1'] = "Peridotite Temperature (℃)"
    currentSheet['O1'] = "Peridotite Temperature Error"
    currentSheet['O2'] = float(
        "{0:.2f}".format(round(rmse_peridotite*1000, 2)))
    currentSheet['P1'] = "Mafic Pressure (GPa)"
    currentSheet['Q1'] = "Mafic Pressure Error"
    currentSheet['Q2'] = float(
        "{0:.2f}".format(round(rmsep_mafic, 2)))
    currentSheet['R1'] = "Mafic Temperature (℃)"
    currentSheet['S1'] = "Mafic Temperature Error"
    currentSheet['S2'] = float(
        "{0:.2f}".format(round(rmse_mafic*1000, 2)))
    currentSheet['T1'] = "Transitional Pressure (GPa)"
    currentSheet['U1'] = "Transitional Pressure Error"
    currentSheet['U2'] = float(
        "{0:.2f}".format(round(rmsep_transitional, 2)))
    currentSheet['V1'] = "Transitional Temperature (℃)"
    currentSheet['W1'] = "Transitional Temperature Error"
    currentSheet['W2'] = float(
        "{0:.2f}".format(round(rmse_transitional*1000, 2)))
    currentSheet['X1'] = "Classification Result"
    # currentSheet['M2'] = listToString(sample_array)
    marker_row = ""
    for i in range(0, len(array_for_peridotite_pressure)):
        current_column = "L"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = float(
            "{0:.2f}".format(round(array_for_peridotite_pressure[i], 2)))
        marker_row = current_row

    for i in range(0, len(array_for_peridotite_temperature)):
        current_column = "N"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = round(
            array_for_peridotite_temperature[i]*1000, 0)
        marker_row = current_row

    for i in range(0, len(array_for_mafic_pressure)):
        current_column = "P"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = float(
            "{0:.2f}".format(round(array_for_mafic_pressure[i], 2)))
        marker_row = current_row

    for i in range(0, len(array_for_mafic_temperature)):
        current_column = "R"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = round(
            array_for_mafic_temperature[i]*1000, 0)
        marker_row = current_row

    for i in range(0, len(array_for_transitional_pressure)):
        current_column = "T"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = float(
            "{0:.2f}".format(round(array_for_transitional_pressure[i], 2)))
        marker_row = current_row

    for i in range(0, len(array_for_transitional_temperature)):
        current_column = "V"
        current_row = str(i+2)
        current_location = current_column + current_row
        currentSheet[current_location] = round(
            array_for_transitional_temperature[i]*1000)
        marker_row = current_row

    for i in range(0, len(nature_data)):
        current_column = "X"
        current_row = str(i+2)
        current_location = current_column + current_row
        if (nature_data[i] == 1):
            classification = 'peridotite'
        elif(nature_data[i] == 2):
            classification = 'transitional'
        else:
            classification = 'mafic'
        #currentSheet[current_location] = nature_data[i]
        currentSheet[current_location] = classification
        marker_row = current_row

    marker_location = "L"+str(int(marker_row)+1)
    theFile.save("./static/result.xlsx")
    return 'a string'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    if os.path.isfile('./file/example.xlsx'):
        os.remove("./file/example.xlsx")
    # if os.path.isfile('./static/result-mafic.xlsx'):
    #     os.remove("./static/result-mafic.xlsx")
    # if os.path.isfile('./static/result-peridotite.xlsx'):
    #     os.remove("./static/result-peridotite.xlsx")
    # if os.path.isfile('./static/result-transitional.xlsx'):
    #     os.remove("./static/result-transitional.xlsx")
    if os.path.isfile('./static/result.xlsx'):
        os.remove("./static/result.xlsx")
    if os.path.isfile('./static/mafic.png'):
        os.remove("./static/mafic.png")
    if os.path.isfile('./static/transitional.png'):
        os.remove("./static/transitional.png")
    if os.path.isfile('./static/peridotite.png'):
        os.remove("./static/peridotite.png")
    if os.path.isfile('./static/first.png'):
        os.remove("./static/first.png")
    if request.method == 'POST':
        input_file = request.files["upload-file"]
        if input_file.filename != '':
            input_file.save("./file/example.xlsx")
            # runpy.run_path(path_name='ANN_classification_web.py')
            run_first_model()
            run_transitional_model()
            run_mafic_model()
            run_peridotite_model()
            run_modify_excel()
    # print(array_for_peridotite)
    return render_template("index.html")


# @app.route('/fig')
# def fig():
#     if os.path.isfile('./file/example.xlsx'):
#         fig = run_first_model()
#         img = BytesIO()
#         fig.savefig(img)
#         img.seek(0)
#         # os.remove("./file/example.xlsx")  # remove Excel from file system after use
#         return send_file(img, mimetype='image/png')
#     return 'a string'


# @app.route('/transitional')
# def transitional():
#     if os.path.isfile('./file/example.xlsx'):
#         fig = run_transitional_model()
#         # run_modify_excel_transitional()
#         img = BytesIO()
#         fig.savefig(img)
#         img.seek(0)
#         # os.remove("./file/example.xlsx")  # remove Excel from file system after use
#         return send_file(img, mimetype='image/png')
#     return 'a string'


# @app.route('/mafic')
# def mafic():
#     if os.path.isfile('./file/example.xlsx'):
#         fig = run_mafic_model()
#         # run_modify_excel()
#         img = BytesIO()
#         fig.savefig(img)
#         img.seek(0)
#         # os.remove("./file/example.xlsx")  # remove Excel from file system after use
#         return send_file(img, mimetype='image/png')
#     return 'a string'


# @app.route('/peridotite')
# def peridotite():
#     if os.path.isfile('./file/example.xlsx'):
#         fig = run_peridotite_model()
#         # run_modify_excel_peridotite()
#         img = BytesIO()
#         fig.savefig(img)
#         img.seek(0)
#         # os.remove("./file/example.xlsx")  # remove Excel from file system after use
#         return send_file(img, mimetype='image/png')
#     return 'a string'


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
    app.run(debug=False)
