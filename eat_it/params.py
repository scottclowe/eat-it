#!/usr/bin/env python

manual_clusters = [
    [14,15,16,17,18,24,25,26,27,30,31,32,33,34,35,36,37],
    [29,3,8,9,10,12,13],
    [1,2,4,5,6,7,11,19,20,21,22,23,28],
    ]

manual_cluster_cols = [['P' + str(i) for i in cluster] for cluster in manual_clusters]

xor_cols = manual_cluster_cols[0]
