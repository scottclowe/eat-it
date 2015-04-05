#!/usr/bin/env python

import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
import sklearn.ensemble
import sklearn.decomposition
import pandas as pd
import numpy as np
import datetime
import csv

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

# Calculate the age of each sample, in days
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
train['Age'] = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in train['Open Date']]
test['Age']  = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in test['Open Date']]

train['logAge'] = np.log(train['Age'])
test['logAge']  = np.log(test['Age'])

train['isBig'] = train['City Group']=='Big Cities'
test['isBig']  =  test['City Group']=='Big Cities'

train['isIL'] = train['Type']=='IL'
test['isIL']  =  test['Type']=='IL'

# Take the log of the age of the venue
#X  = np.log(train[['Age']].values.reshape((train.shape[0],1)))
#Xt = np.log(test[['Age']].values.reshape((test.shape[0],1)))

cols = ['isIL', 
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
        'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20',
        'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
        'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37',
        'isBig', 'Age']

pcols = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
        'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20',
        'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
        'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37']

cols2use = ['isIL', 'isBig', 'Age']

# Drop the string
#train_nostr = train.drop(['Open Date','City','City Group','Type'], 1)

train_pmtx = train.as_matrix(columns=pcols)
test_pmtx  = test.as_matrix(columns=pcols)

compressor = sklearn.decomposition.PCA(n_components=3)
compressor.fit(train_pmtx)
train_PCA = compressor.transform(train_pmtx)
test_PCA = compressor.transform(test_pmtx)

X  = np.concatenate( (train_PCA, train.as_matrix(columns=cols2use)) ,1)
Xt = np.concatenate( (test_PCA, test.as_matrix(columns=cols2use)) ,1)

# The target is of course the revenue
y  = train['revenue'].values

# We will use Linear Regression as our model
#clf = sklearn.linear_model.LinearRegression()
clf = sklearn.ensemble.RandomForestRegressor(1000, random_state=2468)

# Cross-validate
scores=[]
ss = sklearn.cross_validation.KFold(len(y), n_folds=5, shuffle=True, random_state=42)
for trainCV, testCV in ss:
    # Split into train and test
    X_train, X_test, y_train, y_test = X[trainCV], X[testCV], y[trainCV], y[testCV]
    # Fit the classifier
    #clf.fit(X_train, np.log(y_train))
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
    # Predict the revenue
    #y_pred = np.exp(clf.predict(X_test))
    y_pred = clf.predict(X_test)
    # Compute mean squared error
    mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
    print(mse**0.5)
    scores.append(mse)

# Average RMSE from cross validation
scores = np.array(scores)
print("CV Score: {}".format(np.mean(scores**0.5)))

# Fit model again on the full training set
clf.fit(X,np.log(y))
print(clf.feature_importances_)
# Predict test.csv & reverse the log transform
yp = np.exp(clf.predict(Xt))

# Write submission file
refactor = 0.5
type_search = 'MB'
li = test['Type'].values==type_search
print('Found {} of {} are {} types'.format(sum(li),len(li),type_search))
yp[li] = yp[li]*refactor

sub = pd.DataFrame(test['Id'])
sub['Prediction'] = yp
sub.to_csv('RF_PCA_initial_FIXSEED_{}{}.csv'.format(type_search,refactor), index=False)

