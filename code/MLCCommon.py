import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def getB(n):
    if n == 0:
        return(None)
    if n == 1:
        return([[0], [1]])
    else:
        res0 = getB(n - 1)
        res = list()
        for a in res0:
            res.append([0] + a)
        for a in res0:
            res.append([1] + a)
        return(res)
def plotROCCurve(f_t, text, filename):
    """
     Plot a ROC curve from fpr, tpr
    """

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {'text.usetex' : True,
       'font.size' : 15,
#          'font.family' : 'lmodern',
       'text.latex.unicode': True}
    plt.rcParams.update(params) 
    i = 1
    for fpr, tpr, auc in f_t:
        print auc
        label = str(i) + (' AUC = %.2g' % auc)
        plt.plot(fpr, tpr, label = label)
        i += 1
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.tick_params(axis='both', which='major')
    plt.title(str(text), fontsize=15)
    plt.legend()
    if filename != None:
        plt.savefig(filename + ".eps", bbox_inches = 'tight')
    plt.show()
def getShowROC(classifier, Xc, Yc):
        Ys = classifier.decision_function(Xc)
        fpr, tpr, _ = roc_curve(Yc, Ys)
        plotROCCurve(fpr, tpr, "ROC", None)
def plotROC(y_real = None, y_predicted = None):
    r = []
    for y in y_predicted:
        fpr, tpr, _ = roc_curve(y_real, y)
        auc = roc_auc_score(y_real, y)
        r.append((fpr, tpr, auc))
    plotROCCurve(r, "ROC", None)
def getXY(classToTrain, X, Y, badValue = 999):
    """
    Get objects and answers for class classToTrain
    for which answers are available
    """
    
    haveAnswersObjectsIndices = np.where(Y[:, classToTrain] != badValue)
    classX = X[haveAnswersObjectsIndices, :][0]
    classY = Y[haveAnswersObjectsIndices, classToTrain][0]
    return classX, classY
def showROCCurve(fpr, tpr, auc_mean, auc_std, filename):
    """
    Plot a ROC curve from fpr, tpr
    """
    
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {'text.usetex' : True,
          'font.size' : 15,
#          'font.family' : 'lmodern',
          'text.latex.unicode': True}
    plt.rcParams.update(params) 
    plt.plot(fpr, tpr, label = None)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.tick_params(axis='both', which='major')
    auc_mean = '%.2g' % auc_mean
    auc_std = '%.1g' % auc_std
    plt.title(r"AUC $= %s\,\pm\,%s$" % (auc_mean, auc_std), fontsize=15)
    plt.savefig(filename + ".eps", bbox_inches = 'tight')
    plt.show()
