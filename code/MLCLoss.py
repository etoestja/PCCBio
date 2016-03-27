import numpy as np
from sklearn.metrics import zero_one_loss

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
