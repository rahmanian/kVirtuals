#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahmanian
"""

import numpy as np
from scipy.stats import chisquare, chi2_contingency

class FeatureSelection:
    
    def entropy(self,X):
        x_values, x_counts = np.unique(X, return_counts=True)
        N = X.shape[0] if X.size != 1 else 1
        p = np.array(x_counts)/float(N)
        log2_p = np.log2(p)
        hx = -np.sum(p*log2_p)
        return hx, p
    
    def conditional_entropy(self, X, y):
        hx, px = self.entropy(X)
        x_values, x_counts = np.unique(X, return_counts=True)
        hy = np.zeros(len(x_values),dtype=np.float)
        for i, value in enumerate(x_values):
            mask_v = X==value
            hy[i], _ = self.entropy(y[mask_v])
        return np.sum(px*hy), px, hy
    
    def joint_entropy_two(self, X, y):
        hx, px = self.entropy(X)
        hyx, _, _=self.conditional_entropy(X, y)
        return hx + hyx
    
    def joint_entropy_more(self, X):
        if X.shape[1] < 2:
            raise ValueError('Input must be have more than one feature.')
        values, counts = np.unique(X, return_index=False, return_inverse=False, return_counts=True, axis=0)
        prob = counts/X.shape[0]
        return -np.sum(prob*np.log2(prob))
    
    def mutual_information(self, X1, X2):
        hx2,_ = self.entropy(X2)
        hx2x1,_,_ = self.conditional_entropy(X1,X2)
        return hx2 - hx2x1#, hy, hyx
    
    def conditional_mutual_information(self, X, Y, Z):
        """
        compute:
            I(X;Y|Z)=H(X,Z)+H(Y,Z)-H(X,Y,Z)-H(Z)
        """
        Hz,_ = self.entropy(Z)
        Hxz = self.joint_entropy_two(X,Z)
        Hyz = self.joint_entropy_two(Y,Z)
        tmp = np.hstack((X.reshape(-1,1),Y.reshape(-1,1),
                         Z.reshape(-1,1)))
        Hxyz = self.joint_entropy_more(tmp)
        return Hxz + Hyz - Hxyz - Hz
    
    def MSU(self, X):
         X = X.T
         from pyitlib import discrete_random_variable as drv
         joint_H = drv.entropy_joint(X)
         H = [drv.entropy(X[i]) for i in range(X.shape[0])]
         n = len(H)
         f = n/(n-1)
         sum_H = np.sum(H)
         return f*(1-(joint_H/sum_H))

    def symmetrical_uncertainty(self, X1, X2):
        hx1,_ = self.entropy(X1)
        hx2,_ = self.entropy(X2)
        mi = self.mutual_information(X1, X2)
        return 2*mi/(hx1+hx2)

class FilterSupervised(FeatureSelection):
    def __init__(self, Debug=True):
        self.Debug = Debug
        
    def chi_squared(self, X, y):
        N = X.shape[0]
        x_values, x_counts = np.unique(X, return_counts=True)
        y_values, y_counts = np.unique(y, return_counts=True)
#        print(x_values, x_counts)
#        print(y_values, y_counts)
        A = np.zeros((len(x_values),len(y_values)))
        B = np.array(y_counts)
        R = np.array(x_counts)
        chi = 0
        for i, v in enumerate(x_values):
            mask_x = X == v
            for j, b in enumerate(y_values):
                A[i,j] = np.sum(y[mask_x]==b)
        A.astype(np.int16)
        for i, v in enumerate(x_values):
            for j, b in enumerate(y_values):
                tmp = (R[i]*B[j])/N
                chi += ((A[i,j]-tmp)**2)/tmp
        py_chi ,py_p ,py_dof ,_ = chi2_contingency(A)
        return chi, py_chi, py_p, py_dof
    
    def information_gain(self, X, y):
        return self.mutual_information(X,y)
    
    def mRMR(self, X, y, N):
        if N > X.shape[1]:
            N = X.shape[1]
        elif N <= 0:
            return {}
        mi = np.zeros(X.shape[1])
        featureNums = np.array(range(X.shape[1]))
        for i in range(X.shape[1]):
            mi[i] = self.mutual_information(X[:,i], y)
        best = np.argmax(mi)
        S = {best:mi[best]}
        if self.Debug:
            print(f'Add new features: #{best} with value {S[best]}')
        featureNums = featureNums[featureNums != best]
        while len(S) < N:
            tmp = np.zeros(featureNums.shape[0])
            for i,xi in enumerate(featureNums):
                others = 0.0
                for xj in S:
                    others += self.mutual_information(X[:,xi], X[:,xj])
                tmp[i] = mi[xi]-(others/len(S))
            best_indx = np.argmax(tmp)
            best = featureNums[best_indx]
            featureNums = featureNums[featureNums != best]
            S[best] = tmp[np.argmax(tmp)]
            if self.Debug:
                print(f'Add new features: #{best} with value {S[best]}')
        return S
    def general_framework(self, X, y, N, name=None, alpha=0, beta=0):
        """
        compute:
            I(X_k;y)-beta*sum(I(X_j;X_k))+alpha*sum(I(X_j;X_k|y))
        """
        if N > X.shape[1]:
            N = X.shape[1]
        elif N <= 0:
            return {}
        mi = np.zeros(X.shape[1])
        featureNums = np.array(range(X.shape[1]))
        for i in range(X.shape[1]):
            mi[i] = self.mutual_information(X[:,i], y)
        best = np.argmax(mi)
        S = {best:mi[best]}
        if name:
            if self.Debug:
                print(f'Add new features: #{best} with value {S[best]}')
            if name == 'mim':
                alpha = 0
                beta = 0
                mi = mi[mi != mi[best]]
                featureNums = featureNums[featureNums != best]
                while len(S) < N:
                    best_indx = np.argmax(mi)
                    best = featureNums[best_indx]
                    featureNums = featureNums[featureNums != best]
                    S[best] = mi[best_indx]
                    mi = mi[mi != mi[best_indx]]
                    if self.Debug:
                        print(f'Add new features: #{best} with value {S[best]}')
                return S
            if name == 'mifs':
                alpha = 0
            elif name == 'jmi':
                alpha = 1.0 / len(S)
                beta = 1.0 / len(S)
            elif name == 'cmi':
                alpha = 1
                beta = 1
            elif name == 'mrmr':
                alpha = 0
                beta = 1.0 / len(S)
            featureNums = featureNums[featureNums != best]
            while len(S) < N:
                tmp = np.zeros(featureNums.shape[0])
                for i,xi in enumerate(featureNums):
                    redundancy = 0.0
                    condition = 0.0
                    for xj in S:
                        redundancy += self.mutual_information(X[:,xi], X[:,xj])
                        if not (name in ['mim','mifs','mrmr']):
                            condition += self.conditional_mutual_information(X[:,xi], X[:,xj], y)
                    tmp[i] = mi[xi]-beta*redundancy+alpha*condition
                best_indx = np.argmax(tmp)
                best = featureNums[best_indx]
                featureNums = featureNums[featureNums != best]
                S[best] = tmp[np.argmax(tmp)]
                if self.Debug:
                    print(f'Add new features: #{best} with value {S[best]}')
                if name == 'mrmr':
                    beta = 1.0/len(S)
                elif name == 'jmi':
                    alpha = 1.0 / len(S)
                    beta = 1.0 / len(S)
            return S
        else:
            if self.Debug:
                print(f'Add new features: #{best} with value {S[best]}')
            featureNums = featureNums[featureNums != best]
            while len(S) < N:
                tmp = np.zeros(featureNums.shape[0])
                for i,xi in enumerate(featureNums):
                    others = 0.0
                    for xj in S:
                        others += self.mutual_information(X[:,xi], X[:,xj])
                        condition += self.conditional_mutual_information(X[:,xi], X[:,xj], y)
                    tmp[i] = mi[xi]-beta*redundancy+alpha*condition
                best_indx = np.argmax(tmp)
                best = featureNums[best_indx]
                featureNums = featureNums[featureNums != best]
                S[best] = tmp[np.argmax(tmp)]
                if self.Debug:
                    print(f'Add new features: #{best} with value {S[best]}')
            return S

class FilterUnsupervised(FeatureSelection):
    def __init__(self, Debug=True):
        self.Debug = Debug
        
    def PCA_Entropy(self, X, N=10):
        feature_nums = np.array(range(X.shape[1]))
        covx = np.cov(X.T)
        lamdas = np.linalg.eigvals(covx)
        lamdas = lamdas/np.sum(lamdas)
        H_all,_ = self.entropy(lamdas)
        CE = np.zeros(X.shape[1])
        for i in feature_nums:
            reduced_data = X[:,feature_nums != i]
            covx = np.cov(reduced_data.T)
            lamdas = np.linalg.eigvals(covx) if covx.size != 1 else covx
            lamdas = lamdas/np.sum(lamdas)
            H_i,_ = self.entropy(lamdas)
            CE[i] = H_all - H_i
        CE = [(i, CE[i]) for i in range(len(CE))]
        from operator import itemgetter
        CE = sorted(CE, key=itemgetter(1), reverse=True)
        return CE[:N]
    
    def SVD_Entropy(self, X, N=10):
        feature_nums = np.array(range(X.shape[1]))
        _, SV, _ = np.linalg.svd(X)
        SV2 = SV**2
        V = SV2/np.sum(SV2)
        E, _ =self.entropy(V) 
        H_all = E/len(V)
        CE = np.zeros(X.shape[1])
        for i in feature_nums:
            if self.Debug:
                print(f'Leave feature #{i}')
            reduced_data = X[:,feature_nums != i]
            _,SV,_ = np.linalg.svd(reduced_data)
            SV2 = SV**2
            V = SV2/np.sum(SV2)
            H_i,_ = self.entropy(V)
            CE[i] = H_all - (H_i/len(V))
        CE = [(i, CE[i]) for i in range(len(CE))]
        from operator import itemgetter
        CE = sorted(CE, key=itemgetter(1), reverse=True)
        return CE[:N]
