import sklearn
import pandas as pd
import numpy as np
import datetime
import itertools
import copy
import pickle
import sklearn.svm
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
from eat_it import StratifiedPercentileKFold
from eat_it import scalers
from eat_it import params

    
train = pd.read_csv('data/train.csv', encoding="utf-8")
# Add age in days
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
train['Age'] = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in train['Open Date']]
# add size as boolean field
train['isBig'] = train['City Group']=='Big Cities'
# add each of the big cities as boolean field
#train['isIstanbul'] = train['City']=='İstanbul'
#train['isAnkara'] = train['City']=='Ankara'
#train['isIzmir'] = train['City']=='İzmir'
# add boolean field for type
train['isFC'] = train['Type']=='FC'
train['isDT'] = train['Type']=='DT'
train['isMB'] = train['Type']=='MB'
# Note when there is the missing 17 fields
train['missingSource'] = train[params.xor_cols].apply(lambda x: np.all(x==0), axis=1)


with open('data/genuinetestmap.pkl', 'rb') as hf:
    gtm = pickle.load(hf)


gtest = pd.read_csv('data/genuinetest.csv', encoding="utf-8")
# Add age in days
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
gtest['Age'] = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in gtest['Open Date']]
# add size as boolean field
gtest['isBig'] = gtest['City Group']=='Big Cities'
# add each of the big cities as boolean field
#gtest['isIstanbul'] = gtest['City']=='İstanbul'
#gtest['isAnkara'] = gtest['City']=='Ankara'
#gtest['isIzmir'] = gtest['City']=='İzmir'
# add boolean field for type
gtest['isFC'] = gtest['Type']=='FC'
gtest['isDT'] = gtest['Type']=='DT'
gtest['isMB'] = gtest['Type']=='MB'
# Note when there is the missing 17 fields
gtest['missingSource'] = gtest[params.xor_cols].apply(lambda x: np.all(x==0), axis=1)


test = pd.read_csv('data/test.csv', encoding="utf-8")
# Add age in days
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
test['Age'] = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in test['Open Date']]
# add size as boolean field
test['isBig'] = test['City Group']=='Big Cities'
# add each of the big cities as boolean field
#test['isIstanbul'] = test['City']=='İstanbul'
#test['isAnkara'] = test['City']=='Ankara'
#test['isIzmir'] = test['City']=='İzmir'
# add boolean field for type
test['isFC'] = test['Type']=='FC'
test['isDT'] = test['Type']=='DT'
test['isMB'] = test['Type']=='MB'
# Note when there is the missing 17 fields
test['missingSource'] = test[params.xor_cols].apply(lambda x: np.all(x==0), axis=1)


# Merge Test and Train together, without having revenue for all entries
unlabelled_data = pd.concat((train, gtest), ignore_index=True)


#####################################
# Don't use public test revenues
#data = train

# Add known revenues from public test data
gtestrevenue = pd.read_csv('data/genuinetestrevenue.csv', encoding="utf-8")
labelled_test = pd.merge(gtest, gtestrevenue, on='Id')
# Merge all available training data together
data = pd.concat((train, labelled_test), ignore_index=True)
#####################################


# Assemble list of columns
Pcols = ['P'+str(i) for i in range(1,38)]
PMcols = params.xor_cols
PVcols = [i for i in Pcols if i not in params.xor_cols]
Gcols = ['Age']
Ocols = ['isBig','isFC','isDT','isMB']
cols = Pcols + Gcols + Ocols

# Targets
y = data['revenue'].values

X_indices = data['Id'].values
uX_indices = unlabelled_data['Id'].values

index_is_labelled = np.array([i in X_indices for i in uX_indices])
index_is_train    = np.array([i in train['Id'].values for i in uX_indices])

unlabelled_data_nomissing = np.logical_not(unlabelled_data['missingSource'].values)
data_nomissing = np.logical_not(data['missingSource'].values)
test_nomissing = np.logical_not(test['missingSource'].values)

# Other (already one-hot columns) can stay as they are
XO = data.as_matrix(Ocols).astype(np.float)
tXO = test.as_matrix(Ocols).astype(np.float)

# Need to take logs because sometimes Age can't be mapped correctly by BoxCox
u = np.log(unlabelled_data.as_matrix(Gcols).astype(np.float))
d = np.log(data.as_matrix(Gcols).astype(np.float))
t = np.log(test.as_matrix(Gcols).astype(np.float))
s = scalers.BoxCoxScaler().fit(u)
XG = s.transform(d)
tXG = s.transform(t)

# Valid-always columns
u = unlabelled_data.as_matrix(PVcols).astype(np.float)
d = data.as_matrix(PVcols).astype(np.float)
t = test.as_matrix(PVcols).astype(np.float)
s = scalers.BoxCoxScaler().fit(u)
XPV = s.transform(d)
uXPV = s.transform(u)
tXPV = s.transform(t)

# Missing-sometimes columns
u = unlabelled_data.as_matrix(PMcols).astype(np.float)[unlabelled_data_nomissing]
d = data.as_matrix(PMcols).astype(np.float)
t = test.as_matrix(PMcols).astype(np.float)
s = scalers.BoxCoxScaler(known_min=0).fit(u)
XPM = s.transform(d)
uXPM = s.transform(u)
tXPM = s.transform(t)

###############################
# Build model

X_list = []
tX_list = []
cols_ = []

X_list.append(XG)
tX_list.append(tXG)
X_list.append(XO)
tX_list.append(tXO)
cols_ += Gcols
cols_ += Ocols

s = sklearn.decomposition.FastICA(n_components=2, random_state=889, tol=0.000001, max_iter=10000).fit(uXPV)
XDR_ = s.transform(XPV)
tXDR_ = s.transform(tXPV)
PDRcols_ = ['PV_ICA_'+str(i) for i in range(XDR_.shape[1])]
cols_ += PDRcols_

XS2 = sklearn.manifold.MDS(n_components=1, random_state=888).fit_transform(uXPV)
XDR_2 = XS2[index_is_labelled,:]

tXDR_2 = np.zeros((tXPV.shape[0],1))
my_ids = uX_indices[np.logical_not(index_is_train)]
my_XS2 = XS2[np.logical_not(index_is_train),:]
for i,uid in enumerate(my_ids):
    true_ids = gtm[uid]
    for true_id in true_ids:
        tXDR_2[true_id] = my_XS2[i]

PDR2cols_ = ['PV_MDS_'+str(i) for i in range(XS2.shape[1])]
cols_ += PDR2cols_

X_ = np.concatenate([XG, XO, XDR_], axis=1)
tX_ = np.concatenate([tXG, tXO, tXDR_], axis=1)

print(cols_)
print(X_.shape)
print(tX_.shape)

clf = sklearn.linear_model.Lasso()
clf.fit(X_, y)
ty1 = clf.predict(tX_)

#######

X_ = np.concatenate([XG, XO, XDR_2], axis=1)
tX_ = np.concatenate([tXG, tXO, tXDR_2], axis=1)

print(cols_)
print(X_.shape)
print(tX_.shape)

clf = sklearn.linear_model.Lasso()
clf.fit(X_, y)
ty0 = clf.predict(tX_)

###############################

X_list = []
tX_list = []
cols_ = []

X_list.append(XG[data_nomissing,:])
tX_list.append(tXG[test_nomissing,:])
X_list.append(XO[data_nomissing,:])
tX_list.append(tXO[test_nomissing,:])
cols_ += Gcols
cols_ += Ocols

s = sklearn.decomposition.FastICA(n_components=2, random_state=890, tol=0.000001, max_iter=100000).fit(uXPM)
XDR_ = s.transform(XPM[data_nomissing,:])
tXDR_ = s.transform(tXPM[test_nomissing,:])
PDRcols_ = ['PM_ICA_'+str(i) for i in range(XDR_.shape[1])]
X_list.append(XDR_)
tX_list.append(tXDR_)
cols_ += PDRcols_

X_ = np.concatenate(tuple(X_list), axis=1)
tX_ = np.concatenate(tuple(tX_list), axis=1)

print(cols_)
print(X_.shape)
print(tX_.shape)

clf = sklearn.linear_model.Lasso()
clf.fit(X_, y[data_nomissing])
ty2 = clf.predict(tX_)

###############################

# Take geometric mean
ty = (ty0 * ty1)**0.5
ty[test_nomissing] = (ty[test_nomissing] * ty2) ** 0.5
li = np.isnan(ty)
ty[li] = (ty0[li] + ty1[li]) * 0.5

###############################

#####################################
# Overwrite the revenues of known records
uids = gtestrevenue['Id'].values
revs = gtestrevenue['revenue'].values
for uid,rev in zip(uids,revs):
    true_ids = gtm[uid]
    for true_id in true_ids:
        ty[true_id] = np.round(rev)
#####################################

print(sum(np.isnan(ty)))

print(ty1[1095:1100])
print(ty[1095:1100])

sub = pd.DataFrame(test['Id'])
sub['Prediction'] = ty
sub.to_csv('sub_no_rfa_ICA_doubled4_overwrite.csv', index=False)

