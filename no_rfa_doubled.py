import sklearn
import pandas as pd
import numpy as np
import datetime
import itertools
import copy
import sklearn.svm
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
from eat_it import StratifiedPercentileKFold
from eat_it import scalers
from eat_it import params

def do_cv(cv, clf, X, y, y_transform=None):
    if y_transform is None:
        y_transform = lambda x: x
    scores = []
    all_pred = np.zeros((y.shape))
    for train_index, test_index in cv:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit model
        clf.fit(X_train, y_train)
        # Predict scores for test data
        y_pred = y_transform(clf.predict(X_test))
        # Save all the predictions to an array
        all_pred[test_index] = y_pred
        # Compute mean squared error on this test set
        mse = sklearn.metrics.mean_squared_error(y_transform(y_test),y_pred)
        scores.append(mse)
    # Compute MSE across all samples
    all_score = sklearn.metrics.mean_squared_error(y_transform(y), all_pred)**0.5
    scores = np.asarray(scores)**0.5
    return all_score, scores
    
def get_cv_results(clf, X, y, n_folds=10, n_samp=25):
    all_scores = []
    for seed in range(n_samp):
        cv = StratifiedPercentileKFold.StratifiedPercentileKFold(y, n_folds=n_folds, shuffle=True, random_state=seed, shuffle_windows=True)
        this_score, _ = do_cv(cv, clf, X=X, y=y)
        all_scores.append(this_score)
    return np.mean(all_scores), np.std(all_scores)/np.sqrt(n_folds)

def get_mean_cv_score(*args, **kwargs):
    out = get_cv_results(*args, **kwargs)
    return out[0]

### Recursive feature addition
from sklearn.base import clone

def rfa(clf, X, y, n_folds=10, n_samp=25, col_names=None, verbosity=2):
    
    n_features = X.shape[1]
    n_features_to_select = n_features
    step = 1
    
    if col_names is None:
        col_names = range(n_features)
    col_names = np.asarray(col_names)
    
    support_ = np.zeros(n_features, dtype=np.bool)
    ranking_ = n_features * np.ones(n_features, dtype=np.int)
    last_score = None
    
    # Feature addition
    while np.sum(support_) < n_features_to_select:
        # Previously added features
        features_already = np.arange(n_features)[support_]
        # Features to test
        features_to_test = np.arange(n_features)[np.logical_not(support_)]
        
        # Rank the remaining features
        estimator = clone(clf)
        
        #####################################
        # FIT THE CLASSIFIER ON A NESTED FOLD
        #####################################
        
        scores = np.zeros(len(features_to_test))
        for feature_index, test_feature in enumerate(features_to_test):
            features = np.union1d(features_already, [test_feature])
            scores[feature_index] = get_mean_cv_score(estimator, X[:, features], y, n_folds=n_folds, n_samp=n_samp)
            if verbosity>=2:
                print("\tScored %.2f with %s" % (scores[feature_index], ', '.join(col_names[features])))
        
        # Sort the scores in ascending order
        score_order_index = np.argsort(scores)
        ordered_scores   = scores[score_order_index]
        ordered_features = features_to_test[score_order_index]
        
        # Break if no features can improve score
        if last_score is not None and last_score < ordered_scores[0]:
            if verbosity:
                print('No more improvement possible from {} to {} features'.format(
                        len(features_already),len(features_already)+1))
            break
        
        # Only add `step` many features if it doesn't take us past the target
        n_add = min(step, n_features_to_select - np.sum(support_))
        
        # Only add features which don't make performance go down
        if last_score is not None:
            n_add = min(n_add, len(np.nonzero(ordered_scores < last_score)))
        
        # Select best.
        # We will MINIMISE scoring function!!!
        features_to_add = ordered_features[0:n_add]
        for i in range(n_add):
            if verbosity:
                print('Adding feature {} (scored {})'.format(col_names[ordered_features[i]], ordered_scores[i]))
        
        # Add the features
        support_[features_to_add] = True
        ranking_[features_to_add] = np.sum(support_) + 1 + np.arange(features_to_add)
        
        # Update score monitor
        last_score = ordered_scores[0]
        
    if verbosity:
        print("Best score is {} with features:\n\t{}".format(last_score,', '.join(col_names[support_])))
    
    return support_, ranking_
    
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

data = train

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

index_is_labelled = [i in X_indices for i in uX_indices]
index_is_labelled = np.asarray(index_is_labelled)

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

s = sklearn.decomposition.FastICA(n_components=4, random_state=889, tol=0.000001, max_iter=10000).fit(uXPV)
XDR_ = s.transform(XPV)
tXDR_ = s.transform(tXPV)
PDRcols_ = ['PV_ICA_'+str(i) for i in range(XDR_.shape[1])]
X_list.append(XDR_)
tX_list.append(tXDR_)
cols_ += PDRcols_

X_ = np.concatenate(tuple(X_list), axis=1)
tX_ = np.concatenate(tuple(tX_list), axis=1)

print(cols_)
print(X_.shape)
print(tX_.shape)

clf = sklearn.linear_model.ElasticNet()
clf.fit(X_, y)
ty = clf.predict(tX_)

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

s = sklearn.decomposition.FastICA(n_components=4, random_state=890, tol=0.000001, max_iter=100000).fit(uXPM)
XDR_ = s.transform(XPM[data_nomissing,:])
tXDR_ = s.transform(tXPM[test_nomissing,:])
PDRcols_ = ['PV_ICA_'+str(i) for i in range(XDR_.shape[1])]
X_list.append(XDR_)
tX_list.append(tXDR_)
cols_ += PDRcols_

X_ = np.concatenate(tuple(X_list), axis=1)
tX_ = np.concatenate(tuple(tX_list), axis=1)

print(cols_)
print(X_.shape)
print(tX_.shape)

clf = sklearn.linear_model.ElasticNet()
clf.fit(X_, y[data_nomissing])
ty2 = clf.predict(tX_)

###############################

ty[test_nomissing] = 0.5 * (ty[test_nomissing] + ty2)

###############################

sub = pd.DataFrame(test['Id'])
sub['Prediction'] = ty
sub.to_csv('sub_no_rfa_ICA_doubled.csv', index=False)

