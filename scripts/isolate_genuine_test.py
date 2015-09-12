#!/usr/bin/env python

# Import package dependencies
import pandas as pd
import numpy as np
import collections
import datetime
import pickle
import yaml
# Import from this project
from eat_it import utils
from eat_it import params

#########################
verbosity = 1

#########################
def search_cluster(cluster, uhistdict):
    '''
    Given a cluster, find the true entry.
    '''
    cluster_centre = collections.OrderedDict()
    field_name = []
    field_clarity = []
    field_altdist = []
    for col in cluster.columns:
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
        
        # Remember how obvious the field was
        sort_idx = np.argsort(y - uhistdict[col][1])
        field_name.append(col)
        field_clarity.append((y[sort_idx[-1]] - uhistdict[col][1][sort_idx[-1]]) /
                        (y[sort_idx[-2]] - uhistdict[col][1][sort_idx[-2]]))
        field_altdist.append(y[sort_idx[-2]] - uhistdict[col][1][sort_idx[-2]])
    
    return cluster_centre, field_name, field_clarity, field_altdist

def find_matches(cluster, cluster_centre):
    li_centre = np.ones(len(cluster),dtype=bool)
    for col in cluster.columns:
        if col=='Id':
            continue
        li_centre = li_centre * (cluster[col].values == cluster_centre[col])
    return li_centre

def lenient_find_matches(cluster, cluster_centre, clarity_order, field_name):
    li_centre = np.ones(len(cluster),dtype=bool)
    for idx in clarity_order:
        col = field_name[idx]
        if len(np.unique(cluster[li_centre][col].values))==1:
            continue
        li_centre = li_centre * (cluster[col].values == cluster_centre[col])
    return li_centre

def col_xor(cluster_centre):
    T = np.array([cluster_centre[col] for col in params.xor_cols])
    T_xor = any([np.all(T==0), np.all(T!=0)])
    return T_xor

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
    # Start with blank in case we can't manage it
    #genuinetestmap[trigger] = None
    if verbosity>=2:
        print('Considering cluster {}: {}={}'.format(i, trigger_column, trigger))
    # Isolate the cluster + junk which has this opening date
    cluster = test[test[trigger_column].values==trigger]
    if len(cluster) > expected_num_entries*1.5:
        print('Cluster with {}={} is probably two entries'.format(trigger_column, trigger))
    # Find the centre of the cluster
    cluster_centre, field_name, field_clarity, field_altdist = search_cluster(cluster, uhistdict)
    # Now see which entries in the table match the cluster centre
    li_centre = find_matches(cluster, cluster_centre)
    
    # Do some sanity checks
    # Check the cluster centre exists in the test dataset
    if sum(li_centre)==0:
        print('On first try, failed to find the true datapoint with {}={}'.format(trigger_column,
                                                                trigger))
        if len(cluster) > expected_num_entries*1.5:
            # Sanity check:
            if len(cluster) > expected_num_entries*2.5:
                print('There seem to be more than two entries! Code does not handle this.')
                continue
            if len(cluster) > expected_num_entries*1.5:
                print('Cluster with {}={} is indeed two entries'.format(trigger_column, trigger))
            # There are really two completely different entries with the same
            # trigger. So we will separate them out.
            # Look for the most distinctive field to split them on.
            split_idx = np.argsort(field_altdist)[-1]
            split_col = field_name[split_idx]
            
            # Take the histogram of entries in the cluster
            x,y = utils.uniquehist(cluster[split_col].values, uhistdict[split_col][0])
            # Normalise y so it is a distribution
            y = y/sum(y)
            # Sort the indices by distance from expected distribution
            sort_idx = np.argsort(y - uhistdict[split_col][1])
            # Take the two most surprising values
            split_val1 = x[sort_idx[-1]]
            split_val2 = x[sort_idx[-2]]
            if split_val2 < 0:
                # Seems that the two are clones after all?
                print('The two entries seem to be clones. Escaping.')
                continue
            cluster1 = cluster[cluster[split_col].values==split_val1]
            cluster2 = cluster[cluster[split_col].values==split_val2]
            
            # CLUSTER 1
            # Find the centre of the cluster
            cluster_centre, field_name, field_clarity, field_altdist = search_cluster(cluster1, uhistdict)
            # Now see which entries in the table match the cluster centre
            li_centre = find_matches(cluster, cluster_centre)
            # Sanity check
            if sum(li_centre)==0:
                print('Could not find the true entry on first try, cluster 1')
                li_centre = lenient_find_matches(cluster, cluster_centre, np.argsort(field_clarity)[::-1], field_name)
            if sum(li_centre)==0:
                print('Could not find the true entry on second try, cluster 1')
            elif not col_xor(cluster_centre):
                print('Failed XOR test for cluster 1 {}={}'.format(trigger_column, trigger))
            else:
                ids = cluster[li_centre]['Id'].values
                uid = -ids[0]
                genuinetestmap[uid] = ids
                genuinetestdict[uid] = cluster_centre
                genuinetestdict[uid]['Id'] = uid
            
            # CLUSTER 2
            # Find the centre of the cluster
            cluster_centre, field_name, field_clarity, field_altdist = search_cluster(cluster2, uhistdict)
            # Now see which entries in the table match the cluster centre
            li_centre = find_matches(cluster, cluster_centre)
            # Sanity check
            if sum(li_centre)==0:
                print('Could not find the true entry on first try, cluster 2')
                li_centre = lenient_find_matches(cluster, cluster_centre, np.argsort(field_clarity)[::-1], field_name)
            if sum(li_centre)==0:
                print('Could not find the true entry on second try, cluster 2')
            elif not col_xor(cluster_centre):
                print('Failed XOR test for cluster 2 {}={}'.format(trigger_column, trigger))
            else:
                ids = cluster[li_centre]['Id'].values
                uid = -ids[0]
                genuinetestmap[uid] = ids
                genuinetestdict[uid] = cluster_centre
                genuinetestdict[uid]['Id'] = uid
            
            continue
            
        else:
            # Try to find the best match with less restriction
            li_centre = lenient_find_matches(cluster, cluster_centre, np.argsort(field_clarity)[::-1], field_name)
            if sum(li_centre)==0:
                print('On second try, Failed to find the true datapoint with {}={}'.format(trigger_column,
                                                                        trigger))
                print(cluster_centre)
                continue
    
    # Should do XOR sanity check now
    if not col_xor(cluster_centre):
        print('Failed XOR test for {}={}, but using it anyway'.format(trigger_column, trigger))
        print(cluster_centre)
    # Everything seems okay
    elif verbosity>=2:
        print('Successfully detected {} centres for {}={}'.format(sum(li_centre),
                                                            trigger_column, trigger))
    # Add to our holding variables
    ids = cluster[li_centre]['Id'].values
    uid = -ids[0]
    genuinetestmap[uid] = ids
    genuinetestdict[uid] = cluster_centre
    genuinetestdict[uid]['Id'] = uid

# Save the isolated test examples
# Build a pandas table
genuinetest = pd.DataFrame.from_dict(genuinetestdict, orient='index')
# Move Id column to start
cols = list(genuinetest.columns)
cols = cols[-1:] + cols[:-1]
genuinetest = genuinetest[cols]
# Save isolated genuine test examples as CSV
genuinetest.to_csv('data/genuinetest.csv', index=False)

# Save a dictionary of which test entries map to this cluster centre
with open('data/genuinetestmap.pkl', 'wb') as hf:
    pickle.dump(genuinetestmap, hf)

# Save a dictionary of which test entries map to this cluster centre
with open('data/genuinetestmap.txt', 'w') as hf:
    for k,v in genuinetestmap.items():
        hf.write('{}: {}\n'.format(k,v))

