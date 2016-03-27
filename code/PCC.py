class PCC:
    Xc = None
    Yc = None
    C = None
    X = None
    Y = None
    badValue = None
    m = 0
    loss_ = None
    def __init__(self, badValue = 999, loss = 'hamming'):
        self.badValue = badValue
        self.loss = loss
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
            fitOne(self, i)
    def fitOne(self, i):
        self.C[i] = None
        ind = np.where(self.Y[:, i] != self.badValue)
        X0 = self.X[ind]
        Yc = self.Y[ind][:, i]
        Y1 = np.copy(self.Y[ind][:, 0:i])
        
        #print Yc.shape
        #print Y1.shape
        
        #for k in range(Y1.shape[0]):
        #    for j in range(Y1.shape[1]):
        #        if Y1[k, j] == self.badValue:
        #            #print k, j
        #            Y1[k, j] = self.predict(X0[k], j)
        
        if Y1.shape[1] > 0:
            #print X[ind].shape
            #print Y1.shape
            Xc = np.concatenate((self.X[ind], Y1), axis=1)
        else:
            Xc = self.X[ind]
        
        self.Xc[i] = Xc
        self.Yc[i] = Yc
        
        #class_weight = 'balanced'
        tmpLR = LR(fit_intercept = False, solver='liblinear', n_jobs=3)
        tmpLR.fit(Xc, Yc)
        self.C[i] = tmpLR
        self.showFPRTPR(i)
        
        print("i=%d coef=%s" % (i, str(tmpLR.coef_)))
        
    def adjustOne(self, i, fprThresold = 0.2):
        Xc = self.Xc[i]
        Yc = self.Yc[i]
        #getShowROC(self.C[i], Xc, Yc)
        objInd = self.getIndexByFPR(i, fprThresold)
        self.setClassifierObject(i, objInd)
        self.showFPRTPR(i)
    def updatePredictions(self, j):
        for i in range(self.X.shape[0]):
            if self.Y[i][j] == self.badValue:
                self.Y[i][j] = self.predict(X[i], j)
    def showFPRTPR(self, i):
        Xc = self.Xc[i]
        Yc = self.Yc[i]
        classifier = self.C[i]
        N = len(np.where(Yc == 0)[0])
        P = len(np.where(Yc == 1)[0])
        TP = len(np.where((classifier.predict(Xc) == 1) & (Yc == 1))[0])
        FP = len(np.where((classifier.predict(Xc) == 1) & (Yc == 0))[0])
       # print("Got FPR=%.1f %% TPR=%.1f %%" % (100. * FP / N, 100. * TP / P))
    def getIndexByFPR(self, i, fprLow):
        classifier = self.C[i]
        Xc = self.Xc[i]
        Yc = self.Yc[i]
        YScore = classifier.decision_function(Xc)
        Ys = np.argsort(YScore)[::-1]
        N = len(np.where(Yc==0)[0])
        P = len(np.where(Yc==1)[0])
        fpr0 = 0
        tpr0 = 0
        res = None
        for i in range(Ys.shape[0]):
            ansr = Yc[Ys[i]]
            if ansr == 0:
                fpr0 += 1
            if ansr == 1:
                tpr0 += 1
                
            if 1. * fpr0 / N >= fprLow and res == None:
                return(Ys[i])
    def setClassifierObject(self, i, index):
        classifier = self.C[i]
        Xc = self.Xc[i]
        classifier.intercept_ = 0
        classifier.intercept_ = -classifier.decision_function([Xc[index]])
    def probabilityItem(self, x, y):
        res = 1
        for i in range(len(y)):
            x1 = list(x) + y[0:i]
            if self.C[i] != None:
                res *= self.C[i].predict_proba([x1])[0][y[i]]
        return(res)
    def Loss(self, v1, v2):
        r1 = len(np.where(np.array(v1) != np.array(v2))[0])
        return(np.sqrt(r1))
    def predictMinimizeLoss(self, x):
        smin = None
        res = None
        for v in getB(self.m):
            s = 0
            for v1 in getB(self.m):
                s += self.probabilityItem(x, v1) * self.Loss(v1, v)
            if smin == None or smin >= s:
                smin = s
                res = v
        print("RES="+str(res))
        return(res)
    def probabilityClassRecursive(self, x, ic, il = 0):
        if ic == il:
            if self.C[ic] == None:
                return 0
            else:
                return self.C[ic].predict_proba([x])[0][1]
        else:
            fl = self.C[il].predict_proba([x])[0][1]
            return(fl * self.probabilityClassRecursive(list(x) + [1], ic, il + 1) + 
                  (1 - fl) * self.probabilityClassRecursive(list(x) + [0], ic, il + 1))
    def probabilityClasses(self, x):
        res = np.zeros(self.m)
        i = 0
        while i < self.m:
            if self.C[i] != None:
                res[i] = self.probabilityClassRecursive(x, i)
            i += 1
        return(res)
#    def probabilityClassesSlow(self, x):
#        m0 = 0
#        while m0 < self.m:
#            if self.C[m0] == None:
#                break
#            m0 += 1
#        Bm0 = getB(m0)
#        res = np.zeros(self.m)
#        for v in Bm0:
#            for j in range(m0):
#                if v[j] == 1:
#                    res[j] += probabilityItem(self, x, v)
#        return(res)
    def probabilityClassesSubset(self, x):
        m0 = 0
        while m0 < self.m:
            if self.C[m0] == None:
                break
            m0 += 1
        Bm0 = getB(m0)
        res = None
        maxP = 0
        for v in Bm0:
            p = self.probabilityItem(x, v)
            if p > maxP:
                res = v
                maxP = p
        res = res + [0] * (self.m - m0)
        return(res)
    def predictRankLoss(self, x, i):
        return(self.probabilityClassRecursive(x, i) >= 0.5)
    def probabilityNew(self, x, v, i):
        #v1 = v
        #v1[i] = 0
        #p0 = self.probabilityItem(x, v1)
        #v1[i] = 1
        #p1 = self.probabilityItem(x, v1)
        #return(p1 / (p0 + p1))
        #res = self.probabilityItem(x,v)
        #if v[i] == 0:
        #    res = 1 - res
        #return(res)
        #v1 = v[0:i]+[1]
        #return(self.probabilityItem(x,v1))
        #x1 = list(x)
        #for j in range(i + 1):
        #    if self.C[j] != None:
        #        val = self.C[j].predict_proba([x1])[0][1]
        #        ans = [val]
        #        x1 = x1 + ans
        x1 = list(x) + v[0:i]
        val = self.C[i].predict_proba([x1])[0][1]
        return(val)
    def probabilityClassesNew(self, x):
        res = np.zeros(self.m)
        i = 0
        v = self.probabilityClassesSubset(x)
        while i < self.m:
            if self.C[i] != None:
                res[i] = self.probabilityNew(x, v, i)
            i += 1
        return(res)
