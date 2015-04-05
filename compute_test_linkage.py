import sklearn
import pandas as pd
import numpy as np

train = pd.read_csv('../data/train.csv', encoding="utf-8")
test = pd.read_csv('../data/test.csv', encoding="utf-8")

test.columns

X_dist = np.zeros((X.shape[0],X.shape[0]),dtype=np.uint8)
start_time = time.time()
for i in range(X.shape[0]):
    X_dist[:,i] = np.sum(X == np.expand_dims(X[i,:],0), 1)
    if i%1000==0:
        print(i, time.time()-start_time)
