from MLCCommon import getB
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import hamming_loss
from copy import deepcopy
class PCC:
    # saved dataset array
    # item = dataset for i'th classifier
    Xc = None
    Yc = None
    
    # classifiers array
    C = None
    
    # saved dataset
    X = None
    Y = None
    
    # missing value in answers
    badValue = None
    
    # number of labels
    m = 0
    
    # MLC loss function
    loss_ = None
    
    # one-dimensional estimator
    estimator = None
    
    # rearrangement of labels
    permutation = None
    inverse_permutation = None
    
    # constructor
    def __init__(self, badValue = 999, loss = hamming_loss, estimator = None, permutation = None):
        self.badValue = badValue
        self.loss = loss
        self.estimator = estimator
        self.permutation = permutation
        return None
    
    # init arrays
    def initialize(self, X, Y):
        self.m = Y.shape[1]
        
        if type(self.permutation) == type(None):
            self.permutation = np.arange(0, self.m)
            
        self.inverse_permutation = np.argsort(self.permutation)
        
        self.X = np.copy(X)
        self.Y = np.copy(Y[:, self.permutation])
            
        self.Xc = [None] * self.m
        self.Yc = [None] * self.m
        self.C = [None] * self.m
        
    def inversePermute(self, v):
        return v[self.inverse_permutation]
        
    # fitOne for all labels
    def fit(self, X, Y):
        self.initialize(X, Y)
        for i in range(Y.shape[1]):
            self.fitOne(i)
            
    # add answers as features
    # run estimator.fit()
    def fitOne(self, i):
        self.C[i] = None
        ind = np.where(self.Y[:, i] != self.badValue)
        X0 = self.X[ind]
        Yc = self.Y[ind][:, i]
        Y1 = np.copy(self.Y[ind][:, 0:i])
        
        if Y1.shape[1] > 0:
            Xc = np.concatenate((self.X[ind], Y1), axis = 1)
        else:
            Xc = self.X[ind]
        
        self.Xc[i] = Xc
        self.Yc[i] = Yc
        
        tmpLR = deepcopy(self.estimator)
        tmpLR.fit(Xc, Yc)
        self.C[i] = tmpLR
    
    # time: O(2^m)
    def probabilityItem(self, x, y):
        res = 1
        for i in range(len(y)):
            x1 = list(x) + y[0:i]
            if self.C[i] != None:
                res *= self.C[i].predict_proba([x1])[0][y[i]]
        return(res)
    
    # time: O(2^{2m})
    def predictObject(self, x):
        if self.loss == "Hamming":
            res = self.predictHammingLoss(x)
        elif self.loss == "Subset":
            res = self.predictSubsetLoss(x)
        elif self.loss == "Rank":
            res = self.predictMarginals(x)
        else:
            smin = None
            res = None
            for v in getB(self.m):
                s = 0
                for v1 in getB(self.m):
                    s += self.probabilityItem(x, v1) * self.loss(v1, v)
                if smin == None or smin >= s:
                    smin = s
                    res = v
        return(np.array(self.inversePermute(np.array(res))).astype(np.ndarray))
    
    # time: O(2^{2m} * |X|) in general loss case
    # warning: this method uses Bayes optimal decisions
    # for arbitrary loss, which is slow
    # use predict<Metric> for faster prediction
    # or set self.loss to a string variable (see predictObject)
    def predict(self, X, loss = None):
        if not loss == None:
            self.loss = loss
        X = np.array(X)
        if len(X.shape) == 1:
            return(self.predictObject(X))
        elif len(X.shape) == 2:
            res = np.zeros((X.shape[0], self.m))
            for i in range(X.shape[0]):
                res[i] = self.predictObject(X[i])
            return(res)
        else:
            print("Wrong X shape " + str(X.shape))
            
    # time: O(2^m)
    # warning: uses internal label order!
    def predictMarginal(self, x, i, v0 = []):
        # reached required label
        x1 = np.concatenate((np.array(x), np.array(v0)))
        if len(v0) == i:
            return self.C[i].predict_proba([x1])[0][1]
        elif len(v0) < i:
            j = len(v0)
            res = 0
            for v1 in [0, 1]:
                # new vector
                v01 = list(np.copy(v0))
                v01.append(v1)
                p = self.C[j].predict_proba([x1])[0][v1]
                res += p * self.predictMarginal(x, i, v01)
            return res
        
    # time: O(2^m)
    # 1 + 2 + ... + 2^m
    # warning: uses internal label order!
    def predictMarginals(self, x):
        res = [0] * self.m
        for i in range(self.m):
            res[i] = self.predictMarginal(x, i)
        return(np.array(res))
    
    # time: O(2^m)
    # warning: uses internal label order!
    def predictSubsetLoss(self, x):
        # mode of the distribution
        vmax = None
        pmax = 0
        for v in getB(self.m):
            p = self.probabilityItem(x, v)
            if p > pmax:
                pmax = p
                vmax = v
        return vmax
    
    # time: O(2^m)
    # warning: uses internal label order!
    def predictHammingLoss(self, x):
        return self.predictMarginals(x) >= 0.5
    
    # string representation
    def __repr__(self):
        return("PCC X %d x %d Y %d x %d m %d loss = %s" % (self.X.shape[0], self.X.shape[1], self.Y.shape[0], self.Y.shape[1], self.m, self.loss))
    
    # print P(y|x) for given x and all possible y
    # warning: uses internal label order!
    def printDistribution(self, x):
        for v in getB(self.m):
            print "v = " + str(v) + " p = " + str(self.probabilityItem(x, v))
