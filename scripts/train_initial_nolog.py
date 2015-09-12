import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
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

# Take the log of the age of the venue
X  = train[['Age']].values.reshape((train.shape[0],1)) # np.log(
Xt = test[['Age']].values.reshape((test.shape[0],1))

# The target is of course the revenue
y  = train['revenue'].values

# We will use Linear Regression as our model
clf = sklearn.linear_model.LinearRegression()

# Cross-validate
scores=[]
ss = sklearn.cross_validation.KFold(len(y), n_folds=5, shuffle=True, random_state=42)
for trainCV, testCV in ss:
    # Split into train and test
    X_train, X_test, y_train, y_test = X[trainCV], X[testCV], y[trainCV], y[testCV]
    # Fit the classifier
    clf.fit(X_train, y_train) #np.log(y)
    # Predict the revenue
    y_pred = clf.predict(X_test) #np.exp(
    # Compute mean squared error
    mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
    print(mse**0.5)
    scores.append(mse)

# Average RMSE from cross validation
scores = np.array(scores)
print("CV Score: {}".format(np.mean(scores**0.5)))

# Fit model again on the full training set
clf.fit(X,y) #np.log(y)
# Predict test.csv & reverse the log transform
yp = clf.predict(Xt) #np.exp(

# Write submission file
refactor = 0.245705
type_search = 'MB'
li = test['Type'].values==type_search
print('Found {} of {} are {} types'.format(sum(li),len(li),type_search))
yp[li] = yp[li]*refactor

sub = pd.DataFrame(test['Id'])
sub['Prediction'] = yp
sub.to_csv('sub_nolog_{}{}.csv'.format(type_search,refactor), index=False)

