from PCC import PCC
import numpy as np
from math import factorial

class EPCC:
    # array of PCCs
    C = None

    # number of PCCs
    n_estimators = None

    # number of labels
    m = 0

    # constructor
    def __init__(self, badValue = 999, loss = "Rank", estimator = None, n_estimators = 5):
        self.n_estimators = n_estimators
        self.C = [None] * n_estimators
        for i in range(self.n_estimators):
            self.C[i] = PCC(badValue, loss, estimator)

    # set permutations
    def initialize(self, X, Y):
        self.m = Y.shape[1]
        if self.n_estimators > factorial(self.m):
            raise ValueError("EPCC: n_estimators = %d > %d = m!" % (self.n_estimators, factorial(self.m)))
        for i in range(self.n_estimators):
            self.C[i].permutation = np.random.permutation(self.m)
            print self.C[i].permutation

    # fit all n_estimators PCCs
    def fit(self, X, Y):
        self.initialize(X, Y)
        for i in range(self.n_estimators):
            self.C[i].fit(X, Y)

    # get average predictions
    def predict(self, X, loss = None):
        res = None
        for i in range(self.n_estimators):
            r = self.C[i].predict(X, loss = loss)
            print r
            if type(res) == type(None):
                res = r
            else:
                res += r
        print res
        res /= self.n_estimators
        return(res)
