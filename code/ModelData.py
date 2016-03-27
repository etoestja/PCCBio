from scipy.special import expit
import numpy as np
from MLCCommon import getB
import matplotlib.pyplot as plt

R = 1.5
l = 3
m = 500

def P(x, y):
    f1 = expit(x)
    f2 = expit(x - 2 * y[0] + 1)
    f3 = expit(x + 12 * y[0] - 2 * y[1] - 11)
    if y[0] == 0:
        f1 = 1 - f1
    if y[1] == 0:
        f2 = 1 - f2
    if y[2] == 0:
        f3 = 1 - f3
    return(f1 * f2 * f3)

def get(m_ = 500):
    global m
    m = m_
    Xm = 2 * R * (np.random.rand(m) - 0.5)
    Ym = np.zeros((m, l))

    for i in range(Xm.shape[0]):
        # randomness for each object
        u = np.random.rand()
        for v in getB(l):
            tp = P(Xm[i], v)
            if tp >= u:
                Ym[i] = v
                break
            else:
                u -= tp
    Xm = np.array([Xm]).T
    return Xm, Ym

def plotDistribution():
    Xshow = np.arange(-R, R, 0.01)
    plt.figure(figsize=(8, 6))

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    params = {'text.usetex' : True,
       'font.size' : 25,
    #          'font.family' : 'lmodern',
       'text.latex.unicode': True}
    plt.rcParams.update(params)


    plt.xlim([-R, R])
    plt.ylim([0, 1])
    plt.xlabel("$x$", size=25)
    plt.ylabel("$P(y|x)$", size=25)
    for v in getB(l):
        c = v
        if c == [1, 1, 1]:
            c = [0.5, 0.5, 0.5]
        plt.plot(Xshow, P(Xshow, v), color=c, label=str(v))
    plt.legend(loc=2,prop={'size':13}, ncol=4)
    plt.tick_params(axis='both', which='major')
    plt.savefig("ModelData.eps", bbox_inches = 'tight')
    plt.show()

def bestLoss(result, m):
    # metric value
    mmin = None
    mmax = None
    
    #loss in metric value
    lmin = None
    lmax = None
    
    for loss in result:
        metr = result[loss][m]
        if mmax == None or metr > mmax:
            mmax = metr
            lmax = loss
        if mmin == None or metr < mmin:
            mmin = metr
            lmin = loss
    print lmin, mmin
    print lmax, mmax
