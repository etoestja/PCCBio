from MLCCommon import getB
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import hamming_loss
from copy import deepcopy
class PCC:
    Xc = None
    Yc = None
    C = None
    X = None
    Y = None
    badValue = None
    m = 0
    loss_ = None
    estimator = None
    def __init__(self, badValue = 999, loss = hamming_loss, estimator = None):
        self.badValue = badValue
        self.loss = loss
        self.estimator = estimator
        if estimator == None:
            estimator = LR()
        return None
    def initialize(self, X, Y):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.m = Y.shape[1]
        self.Xc = [None] * self.m
        self.Yc = [None] * self.m
        self.C = [None] * self.m
    def fit(self, X, Y):
        self.initialize(X, Y)
        for i in range(Y.shape[1]):
            self.fitOne(i)
    def fitOne(self, i):
        self.C[i] = None
        ind = np.where(self.Y[:, i] != self.badValue)
        X0 = self.X[ind]
        Yc = self.Y[ind][:, i]
        Y1 = np.copy(self.Y[ind][:, 0:i])
        
        if Y1.shape[1] > 0:
            Xc = np.concatenate((self.X[ind], Y1), axis=1)
        else:
            Xc = self.X[ind]
        
        self.Xc[i] = Xc
        self.Yc[i] = Yc
        
        #class_weight = 'balanced'
        # fit_intercept = False
        tmpLR = deepcopy(self.estimator)
        tmpLR.fit(Xc, Yc)
        self.C[i] = tmpLR
        self.showFPRTPR(i)
        
        print("i=%d coef=%s" % (i, str(tmpLR.coef_)))
    def showFPRTPR(self, i):
        Xc = self.Xc[i]
        Yc = self.Yc[i]
        classifier = self.C[i]
        N = len(np.where(Yc == 0)[0])
        P = len(np.where(Yc == 1)[0])
        TP = len(np.where((classifier.predict(Xc) == 1) & (Yc == 1))[0])
        FP = len(np.where((classifier.predict(Xc) == 1) & (Yc == 0))[0])
       # print("Got FPR=%.1f %% TPR=%.1f %%" % (100. * FP / N, 100. * TP / P))
    def probabilityItem(self, x, y):
        res = 1
        for i in range(len(y)):
            x1 = list(x) + y[0:i]
            if self.C[i] != None:
                res *= self.C[i].predict_proba([x1])[0][y[i]]
        return(res)
    def predictObject(self, x):
        smin = None
        res = None
        for v in getB(self.m):
            s = 0
            for v1 in getB(self.m):
                s += self.probabilityItem(x, v1) * self.loss(v1, v)
            if smin == None or smin >= s:
                smin = s
                res = v
        return(res)
    def predict(self, X):
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
    def __repr__(self):
        return("PCC X %d x %d Y %d x %d m %d loss = %s" % (self.X.shape[0], self.X.shape[1], self.Y.shape[0], self.Y.shape[1], self.m, self.loss))
