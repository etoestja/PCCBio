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
