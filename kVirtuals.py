#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahmanian
"""

import numpy as np
import numpy.matlib as mb
import utility as util
import numba
import time
import pandas as pd
import os.path, sys
from scipy.io import arff, loadmat
from FeatureSelection import FilterSupervised, FilterUnsupervised, FeatureSelection
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics

from multiprocessing import Pool, cpu_count, managers
from functools import partial

class MyManager(managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, managers.ArrayProxy)

class featureClustering(FilterUnsupervised, FeatureSelection):
    def __init__(self, Debug=True):
        self.Debug = Debug
        self.name = ''
        self.numClass = 0
        #                      dataset name, type, path  
        self.data_set_names = {'ALLAML':['mat','dataset/microarray/ALLAML.mat'],
                               'colon':['mat','dataset/microarray/colon.mat'],
                               'Embryonal_tumours':['data','dataset/microarray/Embryonal_tumours.csv'],
                               'B-Cell1':['arff','dataset/microarray/B-Cell1.arff'],
                               'B-Cell2':['arff','dataset/microarray/B-Cell2.arff'],
                               'B-Cell3':['arff','dataset/microarray/B-Cell3.arff'],
                               'TOX_171':['mat','dataset/microarray/TOX_171.mat'],
                               'radsen':['data','dataset/microarray/RadiationSensitivity2.csv'],
                               'lung_cancer':['data','dataset/microarray/lung_cancer_13000.csv'],
                               'lung_cancer_203':['xls', 'dataset/lung_cancer_203_12600.xlsx'],
                               'ovarian_253':['xls', 'dataset/ovarian_PBSII_253_15155.xlsx']}
    def load_data(self, data_set='ALLAML', base='dataset/'):
        try:
            [ds_type, ds_path] = self.data_set_names[data_set]   
            self.name = data_set
        except KeyError:
            print(f'Error: Dataset name <{data_set}> is not valid.')
            sys.exit(-1)
        else:
            if ds_type == 'arff':
                data = arff.loadarff(base+ds_path)
                df = pd.DataFrame(data[0])
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                tmp =  (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,:-1].to_numpy()
            elif ds_type == 'mat':
                data = loadmat(base+ds_path)
                self.label = np.copy(data['Y'])
                self.X = np.copy(data['X'])
            elif ds_type == 'data':
                if data_set == 'parkinsons':
                    data = pd.read_csv(base+ds_path, header=0, index_col=0)
                elif data_set == 'breast_cancer':
                    data = pd.read_csv(base+ds_path, header=None, index_col=0)
                elif data_set == 'radsen':
                    data = pd.read_csv(base+ds_path, header=None, sep=' ')
                    data = data.T
                elif data_set == 'Embryonal_tumours':
                    data = pd.read_csv(base+ds_path, header=0)
                else:
                    data = pd.read_csv(base+ds_path, header=None)
                df = pd.DataFrame(data)
                df = df.dropna(axis=1,how='all')
                df = df.fillna(df.mean())
                if data_set == 'parkinsons':
                    tmp = (df['status'].to_numpy())
                elif data_set in ['breast_cancer','lung', 'image', 'radsen', 'lung_cancer']:
                    tmp = (df.iloc[:,0].to_numpy())
                else:
                    tmp = (df.iloc[:,-1].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                if data_set == 'parkinsons':
                    self.X = df.drop('status', inplace=False, axis=1).to_numpy()
                elif data_set in ['lung','lung_cancer','breast_cancer', 'image', 'radsen']:
                    self.X = df.iloc[:,1:].to_numpy()
                else:
                    self.X = df.iloc[:,:-1].to_numpy()
            elif ds_type == 'xls':
                if data_set not in ['lung_cancer_203', 'ovarian_253']:
                    data = pd.read_excel(base+ds_path, header=0, index_col=0, sheet_name='Data')
                else:
                    data = pd.read_excel(base+ds_path, index_col=0)
                df = pd.DataFrame(data)
                print(df.shape)
                tmp = (df.iloc[:,0].to_numpy())
                self.label = LabelEncoder().fit_transform(tmp).reshape(-1,1)
                self.X = df.iloc[:,1:].to_numpy()
            else:
                print(f'Type of dataset <{ds_type}> is not valid.')
                return
            self.numFeatures = self.X.shape[1]
            if isinstance(self.label[0,0],bytes):
                try:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = int(self.label[i,0]) 
                except:
                    for i in range(self.label.shape[0]):
                        self.label[i,0] = self.label[i,0].decode('utf-8')
            self.name = data_set
            tmp = np.unique(self.label, return_index=False, return_inverse=False,
                          return_counts=False)
            self.numClass = len(tmp)
            self.uniqueLabel = tmp[:]
            self.IS = FilterSupervised()
            
    def _binary(self):
        Means = self.X.mean(axis=0)
        M = mb.repmat(Means,np.size(self.X,0),1)
        
        self.BinX = self.X - M
        self.BinX[self.BinX >= 0] = 1
        self.BinX[self.BinX >= 0] = 0
        
    def _discretization(self, k=2):
#        Means = self.X.mean(axis=0)
#        Stds = self.X.std(axis=0)
#        M = mb.repmat(Means,np.size(self.X,0),1)
#        
#        dMean = self.X - M 
#        dMean[dMean>Stds] = 1
#        dMean[dMean<-Stds] = -1
#        a = (dMean<Stds)
#        b = (dMean>-Stds)
#        dMean[a & b] = 0
#        self.TernX = np.copy(dMean)
        kDist = KBinsDiscretizer(k, encode='ordinal', strategy='uniform')
        self.DiscX = kDist.fit_transform(self.X).astype(np.int)
        
    def _remove_useless(self, thresh=0.98, disc = False):
        if disc:
            useless_features = util.check_same_values(self.DiscX, thr=thresh)
            if useless_features.size > 0:
                self.DiscX = np.delete(self.DiscX, useless_features, 1)    
        else:
            useless_features = util.check_same_values(self.X, thr=thresh)
            if useless_features.size > 0:
                self.X = np.delete(self.X, useless_features, 1)

    def preprocess(self, disc=True, k=2, useless=True, thr=0.98, normal=False):
        if normal:
            self.X = (self.X - self.X.min(axis=0))/(self.X.max(axis=0)-self.X.min(axis=0))
        if disc:
            self._discretization(k=k)
        else:
            self.DiscX = np.copy(self.X)
        if useless:
            self._remove_useless(thresh=thr, disc=disc)
        self.numFeatures = self.X.shape[1] if (not disc) else self.DiscX.shape[1]
    
    def NMutInfo(self, path, recalculate=False):
        if self.Debug:
            print(f'Calculate Normalized Mutual Information for {self.name} dataset.')
        if  (not recalculate) and os.path.exists(path):
            self.NMI = np.load(path)
        else:
            import time
            start = time.time()
            self.NMI = np.zeros((self.numFeatures, self.numFeatures))
            for i in range(self.numFeatures):
                if self.Debug and i%300==0:
                    print(f'\t\tFeatures #{i} to {i+300} vs others.')
                for j in range(i+1, self.numFeatures):
                    self.NMI[i,j] = metrics.normalized_mutual_info_score(self.DiscX[:,i], self.DiscX[:,j])
                    self.NMI[j,i] = self.NMI[i,j]
            print(f"NMI time = {time.time()-start} seconds")
            np.save(path, self.NMI)
    
    def _SU(self, arr, ind):
        i,j = ind[0], ind[1]
        arr[i,j] = self.IS.symmetrical_uncertainty(self.DiscX[:,i], self.DiscX[:,j])
        arr[j,i] = arr[i,j]
    
    def _symmetric_uncertainty(self):
        inds = [(i,j) for i in range(self.numFeatures) for j in range(i+1)]
        MM = MyManager()
        MM.start()
        tmp = MM.np_zeros((self.numFeatures,self.numFeatures))
        pool = Pool(cpu_count()-1)
        func = partial(self._SU, tmp)
        _ = pool.map(func, inds)
        rres = np.array(tmp)
        return rres
    
    def symmetric_uncertainty(self, path, recalculate=False, disc=True):
        if self.Debug:
            print(f'Calculate Symmetric Uncertainty for {self.name} dataset.')
        if  (not recalculate) and os.path.exists(path):
            self.SU = np.load(path)
            if self.Debug:
                print(f'SU loaded from file.')
        else:
            import time
            start = time.time()
            self.SU = self._symmetric_uncertainty()
            print(f"SU time = {time.time()-start} seconds")
            np.save(path, self.SU)
            
    def KNN(self, path, ratio=1, recalculate=False, disc=True):
        if self.Debug:
            print(f'Calculate kNN for {name} dataset.')
            start = time.time()
        if  (not recalculate) and os.path.exists(path):
            self.kNN = np.load(path)
        else:
            self.k = int(np.sqrt(self.numFeatures)*ratio)
            self.kNN = util.kNN(self.DiscX, list(range(self.numFeatures)),self.SU, k=self.k+1, disc=disc)
            np.save(path, self.kNN)
        if self.Debug:
            end = time.time()
            print(f'\tTime for kNN on {name} is {end - start} seconds')
    
    def Density_kNN(self, D_kNN_filename=None, base_path='D_kNN', recalculate=False):
        self.k = int(np.sqrt(self.numFeatures))
        if (not recalculate) and D_kNN_filename and os.path.exists(base_path+"/"+D_kNN_filename):
            self.D_kNN = np.load(base_path+"/"+D_kNN_filename)
        else:
            self.D_kNN = np.zeros(self.numFeatures, dtype=np.float)
            for f in range(self.numFeatures):
                sum_of_SU = 0.0
                for neigbor in self.kNN[f,1:]:
                    sum_of_SU += self.SU[f,neigbor]
                self.D_kNN[f] = sum_of_SU/(self.k)
            name = "D_kNN_"+self.name+".npy" if D_kNN_filename == None else D_kNN_filename 
            np.save(f'{base_path}/{name}', self.D_kNN)
        
    def initial_centers(self):
        self.m = 0
        self.maxSU = -10e36
        sorted_D_kNN = np.argsort(-self.D_kNN)   
        
        Fs_ind = np.copy(sorted_D_kNN)
        Fs_ind = Fs_ind.astype(np.int16)
        self.Fc = np.copy(self.DiscX[:,Fs_ind[0]].reshape(self.DiscX.shape[0],1))
        self.Fc_ind = np.array([Fs_ind[0]], dtype=np.int16)
        Fs_ind = np.delete(Fs_ind, np.where(Fs_ind == self.Fc_ind))
        self.m += 1
        
        for fs in Fs_ind:
            tmpMax= []
            for fc in self.Fc_ind:
                if (fc in self.kNN[fs]) or (fs in self.kNN[fc]):
                    break
                tmpMax += [max(self.maxSU, self.SU[fs,fc])]
            else:
                    if fs >= self.DiscX.shape[1]:
                        print(f'\t second ignored: {fs}')
                        continue
                    self.Fc_ind = np.append(self.Fc_ind, [fs])
                    # print(f'***** data size {self.DiscX.shape}, fs {fs} *******')
                    self.Fc = np.hstack([self.Fc, self.DiscX[:,fs].reshape(self.DiscX.shape[0],1)])
                    self.m += 1
                    self.maxSU = np.max(tmpMax)
        print(f'init: m={self.m} with maxSU={self.maxSU}')
        
    def main_loop(self, path, max_feature=100, MAX_ITER=20, recalculate=False):
        self.Fc_ind = self.Fc_ind[:max_feature]
        #Fc_ind = Fs_ind[:]
        self.Fc = self.Fc[:,:max_feature]
        if max_feature>self.Fc.shape[1]:
            max_feature = self.Fc.shape[1]
        
        print('main loop of my algorithm...')
        if (not recalculate) and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            clusters = np.copy(data['clusters'])
            self.Fc = np.copy(data['fc'])
        else:
            FS = FilterSupervised()
            changed = True
            
            itr = 1
            while changed and itr < MAX_ITER:
                print(max_feature, self.Fc.shape[1])
                assert(max_feature == self.Fc.shape[1])
                print(f'Assign samples ...#{itr}, cent={len(self.Fc_ind)}')
                Fs = np.copy(self.DiscX)
                finished = False
                while not finished:
                    clusters = [[None] for i in range(self.Fc.shape[1])]
                    for k, fs in enumerate(Fs.T):
                        if k%100==0:
                            pass
                        SU_centers = np.zeros(self.Fc.shape[1], dtype=np.float)
                        for i, fc in enumerate(self.Fc.T):
                            tmp = FS.symmetrical_uncertainty(fs, fc)
                            SU_centers[i] = tmp
                        j = np.argmax(SU_centers)
                        clusters[j].append(k)
                    else:
                        finished = True
                print('Samples Assigned.')
                for i in range(self.Fc.shape[1]):
                    clusters[i] = clusters[i][1:]
                print(f'Select center of clusters...#{itr}')
                new_Fc = np.copy(self.Fc)
                change_Fc = False
                for i in range(self.Fc.shape[1]):
                    fc = np.copy(self.Fc[:,i]) 
                    tmp_fc = np.copy(fc)
                    for val in range(len(fc)):
                        mi_1 = util.average_mutual_info(fc, clusters[i], val, self.DiscX) 
                        # mi_1 = util.partialMSU(fc, clusters[i], val, self.DiscX) 
                        # pje_1 = util.partial_joint_entropy(fc, Fc, val)
                        fc[val] = 1 - fc[val]
                        mi_2 = util.average_mutual_info(fc, clusters[i], val, self.DiscX) 
                        # mi_2 = util.partialMSU(fc, clusters[i], val, self.DiscX) 
            		 # pje_2 = util.partial_joint_entropy(fc, Fc, val)
			 # if (mi_2 < mi_1 and pje_2 < pje_1) or (mi_2<mi_1 and pje_2>pje_1) or (mi_2>mi_1 and pje_2<pje_1):
			 #   fc[val] = 1-fc[val]
                        fc[val] = fc[val] if mi_2>mi_1 else 1-fc[val] 
                    tmp = np.array(fc).reshape(len(fc),1)
                    new_Fc[:,i] = tmp[:,0]
                    if np.any(fc != tmp_fc):
                        change_Fc = True
                if not change_Fc:
                    changed = False
                self.Fc = new_Fc.copy()
                print('Center of clusters selected.')
                itr += 1
            np.savez(path, clusters=clusters, fc=self.Fc)
        print('main loop of my algorithm finished.')
        
        self.selected_ = []
        Fs = np.copy(self.DiscX)
        for j, fc in enumerate(self.Fc.T):
            if len(clusters[j]) == 0:
                continue
            mi = np.zeros(len(clusters[j]))
            for k,i in enumerate(clusters[j]):
                if i>=Fs.shape[1]:
                    print(f'\t ignored: {i}')
                    continue
                mi[k] = self.mutual_information(fc, Fs[:,i])
            self.selected_.append(clusters[j][np.argmax(mi)])
    
    def main_loop_FSFC(self, path, max_feature=10000, recalculate=False, disc=True, MAX_ITER=20):
        self.Fc_ind = self.Fc_ind[:max_feature]
        self.Fc = self.Fc[:,:max_feature]
        if max_feature>self.Fc.shape[1]:
            max_feature = self.Fc.shape[1]
        
        print('main loop of FSFC algorithm...')
        if  (not recalculate) and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            clusters = np.copy(data['clusters'])
            self.Fc = np.copy(data['fc'])
        else:
            FS = FilterSupervised()
            changed = True
            
            itr = 1
            print(max_feature, self.Fc.shape[1])
            assert(max_feature == self.Fc.shape[1])
            print(f'Assign samples ...#{itr}')
            Fs = np.copy(self.DiscX)
            finished = False
            clusters = [[] for i in range(self.Fc.shape[1])]
            while not finished:
                    if self.m > self.numFeatures:
                        break
                    clusters = [[] for i in range(self.Fc.shape[1])]
                    for k, fs in enumerate(Fs.T):
                        if k in self.Fc_ind:
                            continue
                        j = np.argmax(self.SU[k,self.Fc_ind])
                        if self.SU[k,j]>self.maxSU:
                            clusters[j].append(k)
                        else:
                            self.Fc = np.hstack((self.Fc, fs.reshape(fs.shape[0],1)))
                            self.Fc_ind = np.append(self.Fc_ind, [k]) 
                            Fs = np.delete(Fs, k, axis=1)
                            self.m += 1
                            break
                    else:
                        finished = True                   
            print('Samples Assigned.')
            np.savez(path, clusters=clusters, fc=self.Fc)
        print('main loop of FSFC algorithm finished.')
        
        self.selected_ = []
        Fs = np.copy(self.DiscX)
        for j, fc in enumerate(self.Fc.T):
            if len(clusters[j]) == 0: #and self.maxSU == 1.0:
                continue
            mi = np.zeros(len(clusters[j]))
            for k,i in enumerate(clusters[j]):
                if disc:
                    mi[k] = self.mutual_information(fc, Fs[:,i])
                else:
                    mi[k] = mutual_info_regression(fc.reshape(-1,1), Fs[:,i])
            self.selected_.append(clusters[j][np.argmax(mi)])
        print(f'Selected={len(self.selected_)}, {self.DiscX.shape}, {np.sum(self.selected_)}')
    
    def _learner(self, full=False):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.numClass, init='k-means++', random_state=0)
        if not full:
            X = self.DiscX[:,self.selected_]
        else:
            X = self.DiscX[:,:]
        kmeans.fit(X)
        y_hat = np.copy(kmeans.labels_)
        y_hat = y_hat.reshape(-1,1)
        # print(y_hat.shape, self.label.shape, X.shape)
        return y_hat   
    
    def compute_scores(self):
        from sklearn.metrics import f1_score, adjusted_mutual_info_score
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        y_hat_full = self._learner(full = True)
        y_hat_partial = self._learner(full = False)
        y_hat_full = y_hat_full.reshape(y_hat_full.shape[0],)
        y_hat_partial = y_hat_partial.reshape(y_hat_full.shape[0],)
        
        y_uniq = np.unique(self.label)
        y = np.zeros(self.label.shape[0])
        for i, _y in enumerate(y_uniq):
            tmp = self.label.T == _y
            y[tmp[0]] = i
                
        f_score_full = f1_score(y, y_hat_full, average = 'micro')
        f_score_part = f1_score(y, y_hat_partial,average = 'micro')
        
        ami_full = adjusted_mutual_info_score(y, y_hat_full)
        ami_part = adjusted_mutual_info_score(y, y_hat_partial)
        
        ars_full = adjusted_rand_score(y, y_hat_full)
        ars_part = adjusted_rand_score(y, y_hat_partial)
        
        silh_full = silhouette_score(self.DiscX, y_hat_full)
        silh_part = silhouette_score(self.DiscX[:,self.selected_], y_hat_partial)
                
        
        return f_score_full, f_score_part, ami_full, ami_part, ars_full, ars_part, silh_full, silh_part
    
    def computing_accuracy(self,path, recalculate=False):
        from sklearn import svm
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        y = self.label.reshape(1,-1)[0]
        if isinstance(y[0], int):
            y = y.astype(np.int)            
        knn = KNeighborsClassifier(n_neighbors=3)
        cross = cross_val_score(knn, self.X, y, cv=5, scoring='accuracy')
        knn_all = cross.mean()
        knn_all_std = cross.std()
        print(f'Accuracy(knn) with all features = {knn_all:0.4f}.')
        
        svm_clf = svm.SVC()
        cross = cross_val_score(svm_clf, self.X, y, cv=5, scoring='accuracy')
        svm_all = cross.mean()
        svm_all_std = cross.std()
        print(f'Accuracy(svm) with all features = {svm_all:0.4f}.')

        rf_clf = RandomForestClassifier(max_depth=3)
        cross = cross_val_score(rf_clf, self.X, y, cv=5, scoring='accuracy')
        rf_all = cross.mean()
        rf_all_std = cross.std()
        print(f'Accuracy(RF) with all features = {rf_all:0.4f}.')

        knn_selected = 0.0
        svm_selected = 0.0
        if  (not recalculate) and os.path.exists(path):
            acc = np.load(path)
            knn_all = np.copy(acc['knn_all'])
            knn_selected = np.copy(acc['knn'])
            # knn_selected_std = np.copy(acc['knn_selected_std'])
            svm_all = np.copy(acc['svm_all'])
            # svm_all_std = np.copy(acc['smv_all_std'])
            svm_selected = np.copy(acc['svm'])
            # svm_selected_std = np.copy(acc['svm_selected_std'])
            rf_all = np.copy(acc['rf_all'])
            # svm_all_std = np.copy(acc['smv_all_std'])
            rf_selected = np.copy(acc['rf'])
            # svm_selected_std = np.copy(acc['svm_selected_std'])
        else:
                       
            num_test = 100 if self.numFeatures>100 else self.numFeatures//2
            
            for selected_size in range(len(self.selected_),len(self.selected_)+1):
                #selected_features, index = util.average_redundancy(clusters, selected_size, SU_Mat)
                ss = self.X[:,self.selected_]
                tmp = cross_val_score(knn, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                knn_selected = tmp.mean()
                knn_selected_std = tmp.std()
                print(f'Accuracy(knn) with {selected_size} select features = {knn_selected:0.4f}.')
                
                tmp = cross_val_score(svm_clf, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                svm_selected = tmp.mean()
                svm_selected_std = tmp.std()
                print(f'Accuracy(svm) with {selected_size} select features = {svm_selected:0.4f}.')

                tmp = cross_val_score(rf_clf, ss[:,:selected_size], y, cv=5, scoring='accuracy')
                rf_selected = tmp.mean()
                rf_selected_std = tmp.std()
                print(f'Accuracy(RF) with {selected_size} select features = {rf_selected:0.4f}.')

            np.savez(path, knn_all=knn_all, knn=knn_selected, svm_all=svm_all, svm=svm_selected, rf_all=rf_all, rf=rf_selected)
        
        res = (knn_all, knn_all_std, knn_selected, knn_selected_std)
        res += (svm_all, svm_all_std, svm_selected, svm_selected_std)
        res += (rf_all, rf_all_std, rf_selected, rf_selected_std)
        return res        

if __name__ == '__main__':        
    fc = featureClustering(Debug=True)
    base = '' #dataset/
    names = ['ALLAML','colon','B-Cell1', 'B-Cell2','B-Cell3',
             'TOX_171', 'Embryonal_tumours', 'lung_cancer', 'radsen',
             'lung_cancer_203', 'ovarian_253']
    for name in names:
        if name == 'radsen': # only execute algorithm for 'radsen' dataset, comment this line for running on all dataset
            print(f'loading and preprocessing for {name} dataset...')
            fc.load_data(name, base=base)
            fc.preprocess(disc=True,k=2)
            #print(np.unique(fc.DiscX))
            print(f'{name} dataset with shape={fc.DiscX.shape} loaded.')
            print(f'Computting SU for {name} dataset...')
            fc.symmetric_uncertainty(path=f'SU_Mat/{name}_SU.npy', recalculate=False, 
                                     disc=True)
            print(f'SU for {name} dataset computed.')
            print(f'Computting kNN for {name} dataset...')
            fc.KNN(path=f'KNN/KNN_{name}.npy', recalculate=False, disc=True)
            print(f'kNN for {name} dataset computed.')
            print(f'Computting D_kNN for {name} dataset...')
            fc.Density_kNN(f'D_kNN_{name}.npy',base_path="D_kNN", recalculate=False)
            print(f'D_kNN for {name} dataset computed.')
            print(f'Select initial centers for {name} dataset...')
            fc.initial_centers()
                
            print(f'Initial centers selected.')
            alg_type = 'FSFC' 
            # alg_type = 'MSU'
            print(f'Main Loop of {alg_type} algorithm...')
            max_feat = 100 if fc.numFeatures>100 else fc.numFeatures//2
                 
            if alg_type == 'MSU':
                    fc.main_loop(path=f'clusters/{alg_type}_{name}.npz',
                                 max_feature=fc.numFeatures,recalculate=False, 
                                 MAX_ITER=20)
            else:
                    fc.main_loop_FSFC(path=f'clusters/{alg_type}_{name}.npz',
                                      max_feature=fc.numFeatures, recalculate=True, 
                                      MAX_ITER=20)
            fs_f, fs_p, ami_f, ami_p, ars_f, ars_p, sil_f, sil_p = fc.compute_scores()
            print(f'f_score_full={fs_f:0.4f}, '
                  f'f_score_part={fs_p:0.4f}\n'
                  f'NMI_full    ={ami_f:0.4f}, '
                  f'NMI_part    ={ami_p:0.4f}\n'
                  f'ARI_full    ={ars_f:0.4f}, '
                  f'ARI_part    ={ars_p:0.4f}\n'
                  f'SILH_full   ={sil_f:0.4f}, '
                  f'SILH_part   ={sil_p:0.4f}\n')
            print(fc.selected_)
            with open(f'Accuracy/{alg_type}_accuracy_{name}.txt','w') as f:
                  f.write(f'f_score_full={fs_f:0.4f},'
                      f'f_score_part={fs_p:0.4f}\n'
                      f'NMI_full    ={ami_f:0.4f},'
                      f'NMI_part    ={ami_p:0.4f}\n'
                      f'ARI_full    ={ars_f:0.4f},'
                      f'ARI_part    ={ars_p:0.4f}\n'
                      f'SILH_full   ={sil_f:0.4f},'
                      f'SILH_part   ={sil_p:0.4f}\n'
                      f'features: {fc.selected_}\n'
                      f'n#feature   ={len(fc.selected_)}')
            knn_full, knn_f_std, knn, knn_s_std, svm_full, svm_f_std, svm,svm_s_std, rf_full, rf_f_std, rf,rf_s_std = \
            fc.computing_accuracy(f"Accuracy/{alg_type}_accuracy_{name}.npz", recalculate=True)
            with open(f'Accuracy/{alg_type}_accuracy_{name}.txt','a') as file:
                    file.write('\n')
                    file.write(f'knn_full \t= {knn_full}({knn_f_std})\n'
                               f'knn_selected = {knn}({knn_s_std})\n'
                               f'svm_full \t= {svm_full}({svm_f_std})\n'
                               f'svm_selected = {svm}({svm_s_std})\n'
                               f'RF_full \t= {rf_full}({rf_f_std})\n'
                               f'RF_selected = {rf}({rf_s_std})\n')
        
