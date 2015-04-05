#!/usr/bin/env python

# Import package dependencies
import pandas as pd
import numpy as np
import datetime
import pickle
import collections
# Import from this project
import utils
import params

#########################
# Main
#########################

print('Loading the raw test data')

# Load the 100,000 strong test data
test = pd.read_csv('data/test.csv')

# Add an Age column
end_dt = datetime.datetime.strptime('2015-1-1', "%Y-%m-%d")
test['Age']  = [(end_dt - datetime.datetime.strptime(open_dt, "%m/%d/%Y")).days for open_dt in test['Open Date']]

# Build a dictionary of occurances of each element for each column
uhistdict = {}
for col in test.columns:
    x,y = utils.uniquehist(test[col].values)
    uhistdict[col] = (x,y/sum(y))

# Will use the highest entropy column: Opening Date (equivalent to Age)
# This is almost a unique identifier for the restaurant
trigger_column = 'Age'
unique_triggers = uhistdict[trigger_column][0]

# Note how many entries we expect to see
# If there is more than 50% more than this, there are probably two the same
expected_num_entries = len(test) * np.median(uhistdict[trigger_column][1])

genuinetestmap = collections.OrderedDict()
genuinetestdict = collections.OrderedDict()

for i,trigger in enumerate(unique_triggers):
    #print('Considering cluster {}: {}={}'.format(i, trigger_column, trigger))
    # Isolate the cluster + junk which has this opening date
    cluster = test[test[trigger_column].values==trigger]
    # Find the centre of the cluster
    cluster_centre = collections.OrderedDict()
    for col in test.columns:
        if col=='Id':
            # Skip the unique ID
            continue
        # Take the histogram of entries in the cluster
        x,y = utils.uniquehist(cluster[col].values, uhistdict[col][0])
        # Normalise y so it is a distribution
        y = y/sum(y)
        # Find the index which occurs more than you'd expect by chance
        idx = np.argmax(y - uhistdict[col][1])
        # This value is the value of the true datapoint for this column
        cluster_centre[col] = uhistdict[col][0][idx]
    # Now see which entries in the table match the cluster centre
    li_centre = np.ones(len(cluster),dtype=bool)
    for col in test.columns:
        if col=='Id':
            continue
        li_centre = li_centre * (cluster[col].values == cluster_centre[col])
    # Do some sanity checks
    # Check the cluster centre exists in the test dataset
    if sum(li_centre)==0:
        print('Failed to find the true datapoint with {}={}'.format(trigger_column,
                                                                trigger))
        print(cluster_centre)
        genuinetestmap[trigger] = None
        continue
    # Should do XOR sanity check NOW
    T = np.array([cluster_centre[col] for col in params.xor_cols])
    
    T_xor = any([np.all(T==0), np.all(T!=0)])
    
    if not T_xor:
        print('Failed XOR test for {}={}'.format(trigger_column, trigger))
        print(cluster_centre)
        genuinetestmap[trigger] = None
        continue
    
    # Everything seems okay
    print('Successfully detected {} centres for {}={}'.format(sum(li_centre),
                                                        trigger_column, trigger))
    if len(cluster) > expected_num_entries*1.5:
        print('This cluster is probably two entries')
    genuinetestmap[trigger] = cluster[li_centre]['Id'].values
    genuinetestdict[trigger] = cluster_centre
    genuinetestdict[trigger]['Id'] = trigger

# Save the isolated test examples
genuinetest = pd.DataFrame.from_dict(genuinetestdict, orient='index')
genuinetest.to_csv('data/genuinetest.csv', index=False)

# Save a dictionary of which test entries map to this cluster centre
with open('data/genuinetestmap.pkl', 'wb') as hf:
    pickle.dump(genuinetestmap, hf)

