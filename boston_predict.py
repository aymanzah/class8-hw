#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from argparse import ArgumentParser
from itertools import combinations
from sklearn.datasets import load_boston

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

hist_dir_path='./hist_fig/'
scatter_dir_path='./scatter_fig/'
corr_dir_path='./corr_fig/'
mult_dir_path='./mult_fig/'
pred_dir_path='./pred_fig/'
features_list=[]
target='MEDV'


boston_dataset = load_boston()

data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print (data.head())

#df1=pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
#df2=pd.DataFrame(boston_data.target)
#df3=pd.merge(df1,df2)

data[target] = boston_dataset.target
print (data.head())

for i in range(0,boston_dataset.feature_names.size):
    features_list.append(str(boston_dataset.feature_names[i]))

features_list.append(target)

print (features_list)

#compute summary statistics

print ("*Calculate mean*")
print (np.mean(data))
print ("*Calculate STD*")
print (np.std(data))


#D-Visualize the data, 1-feature (column) at a time

def plot_histogram():

 if not os.path.exists(hist_dir_path):
     os.makedirs(hist_dir_path)

 sns.set() #for a nicer histogram
 
 #loop on columns and generate histograms
 for col in range(len(features_list)):
 
     df=(data.iloc[:,col])
 
     plt.figure()
     plt.hist(df, bins=20)
     plt.xlabel(features_list[col])
     plt.ylabel('occurrence')
     plt.title('Housing Data:{}'.format(features_list[col]))
 
     print ("Generating histogram file for {} in {} dir".format(features_list[col], hist_dir_path))
 
     plt.savefig('{}{}_hist.png'.format(hist_dir_path,features_list[col]))
     #plt.show() #we can uncomment to display all one by one
     plt.close() #avoid warnings


#E) Visualize the data, 2-features (columns) at a time

def plot_scatter_pairs():

 if not os.path.exists(scatter_dir_path):
     os.makedirs(scatter_dir_path)

 #for col in range(len(features_list)):
 for col1 in range(len(features_list)):
     df1=(data.iloc[:,col1])
     col2=1
     for col2 in range((col2+col1), len(features_list)):
         df2=(data.iloc[:,col2])
         #print ("{}/{}".format(features_list[col1],features_list[col2]))
         plt.figure()
         plt.scatter(df1,df2)
         plt.xlabel(features_list[col1])
         plt.ylabel(features_list[col2])
         plt.title('Housing Data:{}/{}'.format(features_list[col1], features_list[col2]))
 
         print ("Generating scatter file for {}/{} pair in {} dir".format(features_list[col1], features_list[col2], scatter_dir_path))
 
         plt.savefig('{}{}_{}_scatter.png'.format(scatter_dir_path,features_list[col1],features_list[col2]))
         #plt.show() #we can uncomment to display all one by one
         plt.close() #to avoid warnings


#G) Pseudocode for an additional type of plot: correlation matrix

def plot_correlation():

 if not os.path.exists(corr_dir_path):
     os.makedirs(corr_dir_path)

 correlations = data.corr()
 fig, ax = plt.subplots(figsize=(len(features_list), len(features_list)))
 cax = ax.matshow(correlations, vmin=-1, vmax=1)
 fig.colorbar(cax)
 
 ax.set_xticks(range(0, len(features_list)))
 ax.set_yticks(range(0, len(features_list)))

 ax.set_xticklabels(features_list)
 ax.set_yticklabels(features_list)

 print ("Generating corrolation file for all features in {} dir".format(corr_dir_path))
 plt.savefig('{}corrolation.png'.format(corr_dir_path))
 #plt.show()
 plt.close()


def plot_multiple_features():

 if not os.path.exists(mult_dir_path):
    os.makedirs(mult_dir_path)

 #a pick of the most important features, all will be compared vs MEDV  (for less combinations)
 features_sublist=['RM', 'LSTAT', 'TAX', 'CRIM', 'DIS', 'PTRATIO', 'INDUS', 'AGE']

 count=0

 for comb in combinations(features_sublist,4):
  
  feature1,feature2,feature3,feature4 = comb

  N = 500
  random_x = np.linspace(0, 1, N)
  random_y = np.random.randn(N)

  df=data.sort_values(by=[target])

  trace1 = go.Scatter(
      x=df.iloc[:,features_list.index(target)],
      y=df.iloc[:,features_list.index(feature1)],
      name = feature1
  )

  trace2 = go.Scatter(
      x=df.iloc[:,features_list.index(target)],
      y=df.iloc[:,features_list.index(feature2)],
      name = feature2
  )

  trace3 = go.Scatter(
      x=df.iloc[:,features_list.index(target)],
      y=df.iloc[:,features_list.index(feature3)],
      name = feature3
  )

  trace4 = go.Scatter(
      x=df.iloc[:,features_list.index(target)],
      y=df.iloc[:,features_list.index(feature4)],
      name = feature4
  )

  layout= go.Layout(
     title= 'Boston Housing Dataset',
     xaxis= dict(
         title= target),
     yaxis= dict(
         title= 'Values'))

  data_t = [trace1, trace2, trace3, trace4]
  fig= go.Figure(data=data_t, layout=layout)

  print ("Creating multi-ft file for {} {} {} {} in {} dir".format(feature1,feature2,feature3,feature4,mult_dir_path))
  file_n='{}{}_{}_{}_{}.html'.format(mult_dir_path,feature1,feature2,feature3,feature4)

  plotly.offline.plot(fig, filename=file_n ,auto_open=False)


#Perform regression

def predict_price():

 if not os.path.exists(pred_dir_path):
     os.makedirs(pred_dir_path)


 #test and train using LSTAT and RM based on correlation results
 X = pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns = ['LSTAT','RM'])
 Y = data[target]

 X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)

 clf = LinearRegression()
 clf.fit(X_train, y_train)

 predicted = clf.predict(X_test)
 expected = y_test
 print ("Performance based on LSTAT and RM data:")
 print ("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))

 plt.scatter(expected, predicted)
 plt.plot([0, 50], [0, 50] , '--k')

 plt.xlabel('Expected Price')
 plt.ylabel('Predicted price')
 plt.title('Boston prices Expected/Predicted')

 print ("Creating prediction figure based on LSTAT and RM in {} dir".format(pred_dir_path))
 plt.savefig('{}predict_price_LSTAT_RM_based.png'.format(pred_dir_path))
 plt.show()


 #test and train using all data
 X_train, X_test, y_train, y_test = train_test_split(boston_dataset["data"], boston_dataset["target"])

 clf = LinearRegression()
 clf.fit(X_train, y_train)

 predicted = clf.predict(X_test)
 expected = y_test
 print ("Performance based on all data:")
 print("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))

 plt.scatter(expected, predicted)
 plt.plot([0, 50], [0, 50] , '--k')

 plt.xlabel('Expected Price')
 plt.ylabel('Predicted price')
 plt.title('Boston prices Expected/Predicted')

 print ("Creating prediction figure based on boston data in {} dir".format(pred_dir_path))
 plt.savefig('{}predict_price.png'.format(pred_dir_path))
 plt.show()


 #Gradient Boosting Regressor

 clf = GradientBoostingRegressor()
 clf.fit(X_train, y_train)

 predicted = clf.predict(X_test)
 expected = y_test

 plt.scatter(expected, predicted)
 plt.plot([0, 50], [0, 50], '--k')

 plt.xlabel('Expected Price')
 plt.ylabel('Predicted price')
 plt.title('Boston prices Expected/Predicted')

 print ("Creating GBR prediction in {} dir".format(pred_dir_path))
 plt.savefig('{}GardBR.png'.format(pred_dir_path))
 plt.show()

plot_histogram()
plot_scatter_pairs()
plot_correlation()
plot_multiple_features()
predict_price()
