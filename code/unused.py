folds = 5
IFolds = sklearn.cross_validation.KFold(n = X.shape[0], n_folds=folds, shuffle=True,
                               random_state=32)
#IFolds = SKF(Y, folds)
maxC = 10

roc_auc = np.zeros((folds, maxC))
fold = 0
for ITrain, ITest in IFolds:
    # get train & test data
    XTrain, YTrain = X[ITrain], Y[ITrain]
    XTest,  YTest  = X[ITest],  Y[ITest]

    Ys = np.zeros(YTest.shape)
    
    algo = PCC()
    algo.initialize(YTrain.shape[1], XTrain, YTrain, 999)
    for i in range(maxC):
        print("fit %d" % i)
        algo.fitOne(i)
        print("adjust %d" % i)
        algo.adjustOne(i, 0.2)
        print("updatePredictions %d" % i)
        algo.updatePredictions(i)
    
    print("calculating ROC...")
    
    for j in range(XTest.shape[0]):
        Ys[j] = algo.probabilityClasses(XTest[j])
    
    for i in range(maxC):
        ind = np.where(YTest[:, i] != 999)
        X1 = XTest[ind]
        Y1 = YTest[ind, i][0]
        Ys0 = Ys[ind, i][0]

        fpr, tpr, _ = roc_curve(Y1, Ys0)
        plotROCCurve(fpr, tpr, "Fold %d Class %d ROC" % (fold + 1, i + 1), None)
        roc_auc[fold][i] = auc(fpr, tpr)
    fold += 1
       
for i in range(maxC):
    auc_mean = np.mean(roc_auc[:, i])
    auc_std = np.std(roc_auc[:, i])
    print("Class %d AUC %.2g +- %.1g" % (i + 1, auc_mean, auc_std))
randomState = 1234
    
def BRShow(X, Y, folds):
    IFolds = sklearn.cross_validation.KFold(n = X.shape[0], n_folds=folds, shuffle=True,
                                   random_state=randomState)
    #IFolds = SKF(Y, folds)
    maxC = 3

    roc_auc = np.zeros((folds, maxC))
    subsetLossArr = np.zeros(folds)
    fold = 0
    for ITrain, ITest in IFolds:
        # get train & test data
        XTrain, YTrain = X[ITrain], Y[ITrain]
        XTest,  YTest  = X[ITest],  Y[ITest]

        subsetLossTemp = np.zeros(YTest.shape)
        
        for i in range(maxC):
            classifier = LR(class_weight = 'balanced')
            classifier.fit(XTrain, YTrain[:,i])

            # get predicted probabilities for class
            YScore = classifier.predict_proba(XTest)[:, 1]
            
            subsetLossTemp[:, i] = (classifier.predict(XTest) == YTest[:, i])

            # calculate fpr and tpr
            fpr, tpr, _ = roc_curve(YTest[:,i], YScore)
            
            plotROCCurve(fpr, tpr, "Fold %d Class %d ROC" % (fold + 1, i + 1), None)
            roc_auc[fold][i] = auc(fpr, tpr)
        subsetLossArr[fold] = 1. * len(np.where(np.all(subsetLossTemp, axis = 1)==0)[0]) / XTest.shape[0]
        fold += 1

    for i in range(maxC):
        auc_mean = np.mean(roc_auc[:, i])
        auc_std = np.std(roc_auc[:, i])
        print("Class %d AUC %.2g +- %.1g" % (i + 1, auc_mean, auc_std))
    subset_mean = np.mean(subsetLossArr)
    subset_std = np.std(subsetLossArr)
    print("SUBSET %.5f %.1f" % (subset_mean, subset_std))
def PCCShow(X, Y, folds, thr = 0.2, do_new = False):
    IFolds = sklearn.cross_validation.KFold(n = X.shape[0], n_folds=folds, shuffle=True,
                                   random_state=randomState)
    #IFolds = SKF(Y, folds)
    maxC = 3

    roc_auc = np.zeros((folds, maxC))
    fold = 0
    subsetLossArr = np.zeros(folds)
    for ITrain, ITest in IFolds:
        # get train & test data
        XTrain, YTrain = X[ITrain], Y[ITrain]
        XTest,  YTest  = X[ITest],  Y[ITest]

        Ys = np.zeros(YTest.shape)

        algo = PCC()
        algo.initialize(YTrain.shape[1], XTrain, YTrain, 999)
        for i in range(maxC):
           # print("fit %d" % i)
            algo.fitOne(i)
          #  print("adjust %d" % i)
           # algo.adjustOne(i, thr)

    #    print("calculating ROC...")

        subsetLoss = 0
        P = np.zeros(YTest.shape[1])
        N = np.zeros(YTest.shape[1])
        FP = np.zeros(YTest.shape[1])
        TP = np.zeros(YTest.shape[1])
        
        for j in range(XTest.shape[0]):
            if do_new:
                Ys[j] = algo.probabilityClassesNew(XTest[j])
            else:
                Ys[j] = algo.probabilityClasses(XTest[j])
            #ans1 = algo.probabilityClassesSubset(XTest[j])
            ans1 = algo.predictMinimizeLoss(XTest[j])
            ans2 = YTest[j]
            if np.any(ans1 != ans2):
                subsetLoss += 1
               # print("Got error j=%d ans1 = %s ans2 = %s" % (j, str(ans1), str(ans2)))
            for i in range(YTest.shape[1]):
                if ans2[i] == 1:
                    P[i] += 1
                else:
                    N[i] += 1
                if ans1[i] == 1 and ans2[i] == 1:
                    TP[i] += 1
                if ans1[i] == 1 and ans2[i] == 0:
                    FP[i] += 1
        for i in range(YTest.shape[1]):
            print("FOLD %d CLASS %d P=%d N=%d FPR=%lf TPR=%lf" % (fold, i, P[i], N[i], 1. * FP[i] / N[i], 1. * TP[i] / P[i]))
                    
        subsetLossArr[fold] = 1. * subsetLoss / XTest.shape[0]
            

        for i in range(maxC):
            ind = np.where(YTest[:, i] != 999)
            X1 = XTest[ind]
            Y1 = YTest[ind, i][0]
            Ys0 = Ys[ind, i][0]

            fpr, tpr, _ = roc_curve(Y1, Ys0)
            #plotROCCurve(fpr, tpr, "Fold %d Class %d ROC" % (fold + 1, i + 1), None)
            roc_auc[fold][i] = auc(fpr, tpr)
            
        Xshow = np.arange(-0.6, 0.6, 0.01)
        plt.figure(figsize=(10, 10))
        plt.xlim([-1, 1])
        plt.ylim([-0.1, 1.05])
        for v in getB(l):
            c = v
            if c == [1, 1, 1]:
                c = [0.5, 0.5, 0.5]
            YShow = np.zeros(Xshow.shape[0])
            for i in range(Xshow.shape[0]):
                item = Xshow[i]
                item = [item]
                YShow[i] = algo.probabilityItem(item, v)
            plt.plot(Xshow, YShow, color=c, label=str(v))
        plt.legend(loc=2,prop={'size':8})
        plt.show()

        Xmr=np.array([Xm]).T
            
        fold += 1

    for i in range(maxC):
        auc_mean = np.mean(roc_auc[:, i])
        auc_std = np.std(roc_auc[:, i])
        print("Class %d AUC %.2g +- %.1g" % (i + 1, auc_mean, auc_std))
    subset_mean = np.mean(subsetLossArr)
    subset_std = np.std(subsetLossArr)
    print("SUBSET %.5f %.1f" % (subset_mean, subset_std))
