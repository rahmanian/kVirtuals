#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahmanian
"""

import numpy as np
import numpy.matlib as mb
import csv
from FeatureSelection import FilterSupervised, FilterUnsupervised, FeatureSelection
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression

def loadData(filename):
    print('Loading and preprocessing data...')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        Data = list(reader)
    Data = np.array(Data)
    Data = Data.transpose();
    X = Data[:,1:] # Only Features
    y = Data[:,0]  # Class labels
    X = X.astype(np.float)
    
    Means = (X.mean(axis=0)).transpose()
    M = mb.repmat(Means,np.size(X,0),1)
    
    Features = X - M
    Features[Features >= 0] = 1
    Features[Features < 0] = 0
    
    numFeatures = np.size(X,1)
    print('Data loaded.')
    return Features, numFeatures, y

# Partial Joint Entropy
def partial_joint_entropy(base, cluster_centers, end):
    FS = FilterUnsupervised()
    y = cluster_centers[:end+1,:]
    t = base[:end+1]
    X = np.column_stack((t, y))
    PJE = FS.joint_entropy_more(X)
    return PJE

# Average Mutual Information
def average_mutual_info(base, cluster_members, end, Features):
    FS = FilterSupervised()
    mi = 0.0
    if len(cluster_members) == 0:
        return 0
    for i in cluster_members:
        mi += FS.mutual_information(base[:end+1], Features[:end+1,i])
    return mi/len(cluster_members)

def partialMSU(base, cluster_members, end, Features):
    FS = FeatureSelection()
    y = Features[:end+1,:]
    t = base[:end+1]
    XX = np.column_stack((t, y))
    PMSU = FS.MSU(XX)
    return PMSU

# Average Redundancy function.
def average_redundancy(clusters, selected_size, SU_Mat):
    clusters_size = np.zeros(len(clusters), dtype=np.int16)
    selected_features = np.zeros(selected_size, dtype=np.int16) 
    for i, cl in enumerate(clusters):
        clusters_size[i] = len(cl)
    index = np.argsort(-clusters_size)
    for i in range(selected_features.shape[0]):
        AR = np.zeros(len(clusters[index[i]]))
        for j in range(len(clusters[index[i]])):
            others = list(range(0,j))+list(range(j+1,len(clusters[index[i]])))
            for k in others:
                AR[j] += SU_Mat[j,k]
            AR[j] = AR[j]/len(clusters[index[i]])
        select = np.argmax(AR)
        selected_features[i] = clusters[index[i]][select]
    return selected_features, index

def SU_dist(x, y):
    global SU
    if SU is None:
        FS = FilterSupervised()
        su = FS.symmetrical_uncertainty(x,y)
        #t = su if su > 0 else 10e-15
        return 1.0/(su+10e-15)
    else:
        if type(x) is np.ndarray:
            x = int(x[0])
            y = int(y[0])
        # print(f'x={x}, y={y}')
        su = SU[x,y]
        #t = su if su > 0 else 10e-15
        return 1.0/(su+10e-15)

def MI_dist(x, y):
    su = mutual_info_regression(x.reshape(-1,1), y)
    return 1.0/(su+10e-15)

SU = None
def kNN(X, f, _SU = None,k=3, disc = True):
    global SU
    SU = _SU
    if disc:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=SU_dist)
    else:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=MI_dist)
    
    n,m = X.shape
    indx = np.array(f, dtype=int)
    nbrs.fit(indx.reshape(-1,1))
    distances, indices = nbrs.kneighbors(np.array(indx).reshape(-1,1))
    return indices

def check_same_values(X, thr=0.98):
    features = []
    m, n = X.shape
    for i in range(n):
        v, c = np.unique(X[:,i], return_index=False, return_inverse=False, return_counts=True)
        r = c/np.sum(c)
        if np.sum(r>thr)>0:
            features.append(i)
    return np.array(features)
