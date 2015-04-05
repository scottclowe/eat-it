#!/usr/bin/env python

import pandas as pd
import numpy as np
import datetime
import csv


def generate_hack(ids, factor=0.75):
    
    sub = pd.read_csv('submissions/sampleSubmission.csv')
    
    yp = sub['Prediction'].values
    yp[ids] = factor * yp[ids]
    sub['Prediction'] = yp
    
    sub.to_csv('submissions/samp_{}_on_{}.csv'.format(factor,ids),index=False)


def uniquehist(x, bins=None):
    if bins is not None:
        ux = bins
    else:
        ux = np.unique(x)
    count = np.zeros(len(ux),dtype=int)
    for ix,xx in enumerate(ux):
        count[ix] = np.sum(x==xx)
    return ux, count


