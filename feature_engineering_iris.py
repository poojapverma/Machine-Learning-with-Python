# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### import libraries
import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from matplotlib import pyplot as plt

import seaborn as sns

#### import dataset from load_iris function of datasets class

iris_data= load_iris()

### creating a Pandas dataframe from iris dataset

iris=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)

## addting target column to the iris dataframe

iris['Species']=iris_data.target

### print shape of observations

print('no of observations:',iris.shape[0])

print('No of features:',iris.shape[1])

print('info:',iris.info())

print ('summary : \n',iris.describe())

##check for null values

print('Null check:\n', iris.isnull().sum())


#fig, axs = plt.subplots(ncols=3)
'''pairplot  - this function will create a grid of Axes such that each numeric
    variable in ``data`` will by shared in the y-axis across a single row and
    in the x-axis across a single column. The diagonal Axes are treated
    differently, drawing a plot to show the univariate distribution of the data
    for the variable in that column.
'''
#sns.pairplot(iris,hue='Species',palette='Dark2')

#sns.countplot(iris.Species,ax=axs[0])
#sns.countplot(iris.Species)

X=iris.iloc[:,0:4].values
y=iris['Species']

#Split the data into train and test data
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=8)

#scaling the data into standard form ( Normal distribution with mean=0 and variance=1) using StandScaler

sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)

X_test=sc.fit_transform(X_test)
#or use X_train=sc.fit_transform(X_train)
##check for mean and var of transformed data

print("Train data mean= %f , variance=%f" % (abs(X_train.mean()), X_train.var()))

print("Test data mean= %f , variance=%f" % (abs(X_test.mean()), X_test.var()))

#performing LDA

lda=LDA(n_components=1)

X_train_lda=lda.fit_transform(X_train,y_train)

X_test_lda=lda.transform(X_test)

print('training data shape before and after applying lda, before=',X_train.shape,' after = ', X_train_lda.shape)

print('Explained data variance with LDA:%f' %(lda.explained_variance_ratio_*100))

## for more than one components
#print('Explained data variance with LDA:' ,(lda.explained_variance_ratio_.sum()*100))

rc= RandomForestClassifier(max_depth=2, random_state=8)

rc.fit(X_train,y_train)

y_pred=rc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print( 'Confusion Matrix :\n', cm)

print('accuracy score: ', accuracy_score( y_test, y_pred))

'''
rc.fit(X_train_lda,y_train)

y_pred=rc.predict(X_test_lda)

cm=confusion_matrix(y_test,y_pred)

print( 'Confusion Matrix :\n', cm)

print('accuracy score: ', accuracy_score( y_test, y_pred)*100)
'''
sns.set()
#sns.heatmap(cm,cmap='Blues_r')

#print("classification report : \n",classification_report(y_test, y_pred))


#### Principle component Analysis

pca= PCA(n_components=1)

pca.fit(X_train)

X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

print('training data shape before and after applying pca, before=',X_train.shape,' after = ', X_train_pca.shape)

print(f"PCA explained variance ratio : {(pca.explained_variance_ratio_.sum()) * 100}")

rc2= RandomForestClassifier(max_depth=2, random_state=0)

rc2.fit(X_train,y_train)

y_pred_pca=rc2.predict(X_test)

cm_pca=confusion_matrix(y_test,y_pred_pca)

print( 'Confusion Matrix (PCA) :\n', cm_pca)

print('accuracy score (PCA): ', accuracy_score( y_test, y_pred_pca)*100)
