#!/usr/bin/env python

import sklearn
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.externals import joblib

# MAIN #

fname = '/media/scott/scratch/tfi_test_linkage_dist.pkl'

print('Loading test data')
test = pd.read_csv('data/test.csv', encoding="utf-8")

# Don't need to compare Id column - this is trivially always different
test = test.drop('Id', 1)
print(test.columns)

# Convert to matrix
X = test.as_matrix()

# Initialise output matrix. Make it uint8 to take far less space.
X_dist = np.zeros((X.shape[0],X.shape[0]),dtype=np.uint8)
start_time = time.time()
# Wasteful as it does every pair twice. But this is unimportant.
# Only needs to run once.
# Distance is the number of fields which do not match.
for i in range(X.shape[0]):
    X_dist[:,i] = np.sum(X == np.expand_dims(X[i,:],0), 1)
    if i%1000==0:
        print('{} of {} at {} seconds'.format(i, X.shape[0], time.time()-start_time))

# Save the similarity/distance matrix to disk. Takes 12GB.
print('Saving to ' + fname)
joblib.dump(X_dist, fname)

print('Done. Total time = {} seconds'.format(time.time()-start_time))
