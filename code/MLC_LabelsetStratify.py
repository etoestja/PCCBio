import numpy as np
from sklearn.cross_validation import KFold

def LabelsetKFold(Y, n_folds = 3, random_state = None, shuffle = True, force = False):
    result = [None] * n_folds
    for i in range(n_folds):
        result[i] = [[],[]]
    sets = np.vstack({tuple(row) for row in Y})
    for s in sets:
        rows = np.where(np.all(Y == s, axis = 1))[0]
        n = len(rows)
        print n
        
        # Handling low objects per label count case
        n_folds_ = n_folds
        if force and n_folds > n:
            n_folds_ = n
        elif not force and n_folds > n:
            raise ValueError("LabelsetKFold: Force == False and n_folds = %d > %d = n. \
Set Force to True or reconsider your data. Labelset = %s" % (n_folds, n, str(s)))
            
        # Adding labels of size 1 to train
        # to each fold
        if force and n == 1:
            for i in range(n_folds):
                result[i][0].extend(rows)
            continue
        elif not force and n == 1:
            raise ValueError("LabelsetKFold: Force == False and n == 1. Set Force to True or reconsider your data. Labelset = %s" % (str(s)))
        
        folds = KFold(n_folds = n_folds_, n = n, random_state = random_state, shuffle = shuffle)
        
        i = 0
        while i < n_folds:
            for train, test in folds:
                result[i][0].extend(rows[train])
                result[i][1].extend(rows[test])
                i += 1
                if i >= n_folds:
                    break
    return result