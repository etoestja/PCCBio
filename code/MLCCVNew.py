import numpy as np
def processFold(XTrain, YTrain, XTest, YTest, estimator, metrics, do_train):
    estimator.fit(XTrain, YTrain)
    if do_train:
        Yrtrain = estimator.predict(XTrain)
    if XTest.size > 0:
        Yr = estimator.predict(XTest)
    result = {}
    for m in metrics:
        if do_train:
            result[m + ' train'] = metrics[m](Yrtrain, YTrain)
        if XTest.size > 0:
            result[m + ' test'] = metrics[m](Yr, YTest)
        #print m, result[m]
        
    return(result)
def cross_val_score_multiple_metrics(estimator, metrics, cv, X, Y, do_train = False):
    result = {}
    
    for m in metrics:
        result[m + ' train'] = []
        result[m + ' test'] = []
    
    for ITrain, ITest in cv:
        XTrain, YTrain = X[ITrain], Y[ITrain]
        XTest, YTest = X[ITest], Y[ITest]
        rfold = processFold(XTrain, YTrain, XTest, YTest, estimator, metrics, do_train)
        for m in rfold:
            result[m].append(rfold[m])
    return result
def metric_ith_column(x, y, f, i):
    '''
    Given function f(x,y),
    x, y matrices
    Make a function f_(x[:,i],y[:,i])
    '''
    x = np.array(x)
    y = np.array(y)
    a = x[:, i]
    b = y[:, i]
    
#    print "using" +str((a, b))
    try:
        res = f(a,b)
    except:
        res = 0
    return(res)
