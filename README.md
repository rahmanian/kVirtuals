# Python implementation of kVirtuals

This repository contains code for a gene selection method based on kVirtuals: [An unsupervised gene selection method based on multivariate normalized
mutual information of genes](https://www.sciencedirect.com/science/article/abs/pii/S0169743922000235) implemented in Python.

## Requirements

* Python 3.x
* sklearn
* scipy
* multiprocessing
* numpy
* csv

## Executing program

Before running the algorithm,

* You should have five folders with following names: 
  * Accuracy, 
  * Clusters, 
  * SU, 
  * KNN, 
  * and DKNN
* And a Dataset folder that contains data files at cvs, xls, arff, and/or mat formats.
```
python3 kVirtuals.py
```

## Authors

[Mohsen Rahmanian](http://mrahmanian.ir)

## Version History

* 1.0
    * Initial Release
