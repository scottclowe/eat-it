import sklearn
import pandas as pd
import numpy as np
import datetime

from IPython.display import HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import sklearn.linear_model
import sklearn.cross_validation

from eat_it import StratifiedPercentileKFold
from eat_it import boxcoxscaler


train = pd.read_csv('data/train.csv', encoding="utf-8")

# Add age in days
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
train['Age'] = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in train['Open Date']]

# add size as boolean
train['isBig'] = train['City Group']=='Big Cities'

train['isIL'] = train['Type']=='IL'

cols = ['P'+str(i) for i in range(1,38)]
cols.append('Age')
cols.append('isBig')
cols.append('isIL')

X = train.as_matrix(cols)
X = X.astype(np.float)
y = train['revenue'].values


