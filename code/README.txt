Author: Sergey Volodin, 2016
sergei.volodin@phystech.edu
MIT License

What is required?
* Juputer Notebook
* Pandas, Numpy, Scipy, Sklearn

What it does?
* Training {PCC, BR} with LogisticRegression as an estimator
* Using different loss functions for PCC for predictoin
* Comparing results using different metrics
* Printing results

Model data and real data is used.

Files:
* MLC_PCC.ipynb -- main file
* MLCLoss.py -- loss functions for MLC problem
* MLCCommon.py -- common functions used in MLC_PCC.ipynb
* MLCCV.py -- helpers for cross validation in MLC
* ModelData.py -- a set of functions used to generate and show model data described in the article
* PCC.py -- an implementation of Probabilistic Classifier Chains
* Losses.ipynb -- a notebook demonstrating loss functions
