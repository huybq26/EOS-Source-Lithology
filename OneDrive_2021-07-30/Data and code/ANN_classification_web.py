#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:18:41 2021

@author: lilucheng
"""
#this classification model for web used
#inputs: data from the Excel files
#outputs: three labels(Peridotite, transitional, mafic)




import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from numpy import *
from sklearn.metrics import mean_squared_error 


#-------------------------------------------------------
# Read the file  dataframe
data = pd.read_excel('data2_check.xlsx',header = None,skipfooter= 1,index_col=1)

#change into the data we need float
#train data determined from dataframe
Traindata = np.zeros((915,10))  
for i in range(0,915):  
 for j in range(0,10):  
   Traindata[i][j] = data.iloc[i+1,j+6]  


#change nan into 0
for i in range(0,915):  
 for j in range(0,10):  
  if (np.isnan(Traindata[i][j])):
      Traindata[i][j]= 0
  
   
   

#lable from dataframe
Group=np.zeros((915,1))
for i in range(0,915):   
   Group[i] = data.iloc[i+1,24]  


#-------------------------------------------------------
X = Traindata
y = Group



#D=X
#idq=np.where((D[:,0]<30) | (D[:,0]>65))  
#idq0=idq[0]
#newX=np.delete(D,idq0,0)

#newy=np.delete(y,idq0)
#newy=newy.reshape(-1,1)

newX=X
newy=y

X_train, X_test, y_train, y_test = train_test_split(newX, newy, train_size=0.8, random_state = 0)


clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1)


clf = clf.fit(X_train, y_train)




accuracy_ANN = clf.score(X_test, y_test)

print('Accuracy Neural network test:', accuracy_ANN)


#visilize
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
accuracy_ANNtrain = clf.score(X_train, y_train)

print('Accuracy Neural network of train:', accuracy_ANNtrain)



y_result=clf.predict(newX) 
y_result=y_result.reshape(-1,1) 
difference_total=np.zeros((915,1))
difference_total=newy-y_result
idx=np.where(difference_total==0)  
idx0=idx[0]
Same_alldata=len(idx0)/len(y_result)
print('Accuracy Neural network of all data:', Same_alldata)



#------------------------------------------------------------------------------------------
#read natural case data   Hawoii volcano

data2 = pd.read_excel('natural_case_noref.xlsx',header = None,skipfooter= 1,index_col=1)

#train data determined from dataframe
Naturedata = np.zeros((763,10))  
for i in range(0,763):  
 for j in range(0,10):  
   Naturedata[i][j] = data2.iloc[i+1,j+1]  
   
Naturedata_result=clf.predict(Naturedata)

Naturedata_result=Naturedata_result.reshape(-1,1)

#------
#result compare
Previousresult=np.zeros((763,1)) 
for i in range(0,763):   
   Previousresult[i] = data2.iloc[i+1,14]  

difference_total=np.zeros((763,1))
difference_total=Previousresult-Naturedata_result


idx=np.where(difference_total==0)  
idx0=idx[0]
Same_rate=len(idx0)/len(Previousresult)
print('Same number of all the data',Same_rate)   



###chose natural_results by ANN== 1,2,3


idxd1=np.where( (Naturedata_result==1)) 
idxd10=idxd1[0]

idxd2=np.where((Naturedata_result==2)) 
idxd20=idxd2[0]

#idxd3=(np.where(difference_total!=0) and np.where(Naturedata_result==3)) 

idxd3=np.where((Naturedata_result==3)) 
idxd30=idxd3[0]


#########



###show the results by figures
MgO=np.zeros((763,1))
for i in range(0,763):   
   MgO[i] = data2.iloc[i+1,7] 
   

Mg=np.zeros((763,1))
for i in range(0,763):   
   Mg[i] = data2.iloc[i+1,16] 
   
   
cati=np.zeros((763,1))
for i in range(0,763):   
   cati[i] = data2.iloc[i+1,17] 
   
   
sica=np.zeros((763,1)) 
for i in range(0,763):   
   sica[i] = data2.iloc[i+1,18] 



fcmsd=np.zeros((763,1))
for i in range(0,763):   
   fcmsd[i] = data2.iloc[i+1,19] 


###############################
plt.figure()


xx=[0.38,0.92]
yy1=[0.37,0.37]
yy2=[0.05,0.05]
plt.plot(xx,yy1,'b--',alpha=0.3,linewidth=0.5)
plt.plot(xx,yy2,'b--',alpha=0.3,linewidth=0.5)

