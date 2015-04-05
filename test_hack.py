import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
import pandas as pd
import numpy as np
import datetime
import csv

factor = 0.75

sub = pd.read_csv('submissions/sampleSubmission.csv')

ids = [2244,90048, 2679,6347]   # Rarest and youngest
ids = [2679,6347]               # Youngest. Should be able to get it now, but will need to check.
ids = [84446, 6212,17479,47342] # Second youngest and oldest. Not in.
ids = [29949,64895,83582, 11153,62370,74757,75865,89385,99604] # Third youngest and Second oldest
ids = [ 2134, 13062, 36251, 70575, 86590, 95607,  2141, 41852, 64628, 67454, 94768] # 4th young, 3rd oldest

yp = sub['Prediction'].values
yp[ids] = factor * yp[ids]

sub['Prediction'] = yp

sub.to_csv('samp_{}_on_{}.csv'.format(factor,ids),index=False)

