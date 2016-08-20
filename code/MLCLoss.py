import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score, recall_score

def subsetLoss(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    if len(v1.shape) == 1:
        res = np.any(v1 != v2)
    else:
        res = np.sum(np.any(v1 != v2, axis = 1))
    return int(res)

def subsetLossN(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    r = subsetLoss(v1, v2)
    return 1. * r / v1.shape[0]

def HammingLoss(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.sum(v1 != v2)

def FMeasureInverse(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    precision = precision_score(v1, v2)
    recall = recall_score(v1, v2)
    res = 2. * precision * recall / (precision + recall)
    return(1 - res)

def HammingLossN(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    r = HammingLoss(v1, v2)
    return 1. * r / v1.size

def HammingLossClasses(A, B):
    A, B = np.array(A), np.array(B)
    return np.sum(A != B, axis = 0)

def HammingLossObjects(A, B):
    A, B = np.array(A), np.array(B)
    if len(A.shape) == 2:
        return np.sum(A != B, axis = 1)
    else:
        return np.sum(A != B)

def middleLoss(v1, v2, t):
    v1, v2 = np.array(v1), np.array(v2)
    t = np.array(t)
    r = HammingLossObjects(v1, v2)
    r1 = t[r]
    return np.sum(r1)

def FBetaLossClasses(v1, v2, beta = 1):
    v1, v2 = np.array(v1), np.array(v2)
    b = beta ** 2.
    P = precision_score(v1, v2)
    R = recall_score(v1, v2)
    z = (b * P + R)
    if z == 0:
        return np.nan
    r = (1. + b) * P * R / z
    return 1 - r

def FBetaLoss(A, B, beta = 1):
    A, B = np.array(A), np.array(B)
    if len(A.shape) == 2:
        arr = []
        for i in range(A.shape[0]):
            arr.append(FBetaLossClasses(A[i], B[i], beta))
        return np.nanmean(arr)
    else:
        return FBetaLossClasses(A, B, beta)

best_is_min = {"H": True, "P": False, "R": False, "A": False, "S": True}

def bestLoss(result, m):
    # metric value
    mmin = None
    mmax = None

    #loss in metric value
    lmin = None
    lmax = None

    for loss in result:
        metr = np.mean(result[loss][m])
        if mmax == None or metr > mmax:
            mmax = metr
            lmax = loss
        if mmin == None or metr < mmin:
            mmin = metr
            lmin = loss
    if best_is_min[m[0]]:
        print "min", m, lmin, mmin
    else:
        print "max", m, lmax, mmax
