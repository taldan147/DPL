import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

def SGD(X, w, c, epoch):
    norms = [LA.norm(c)]
    lr = 0.01
    batch = 2
    for i in range(epoch):
        perm = np.random.permutation(len(X))
        # lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[indx, :]
            currc = c[indx]
            grad = 1/batch * (currX.transpose() @ ((currX @ w) - currc)) + 0.01*w
            w = w-lr*grad
        norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
    return w, norms

def test_SGD_LS_2():
    X = np.asarray([[-1,-1,1], [1,3,3],[-1,-1,5],[1,3,7]])
    y = np.asarray([0,23,15,39])
    w = np.zeros(3)
    # w = np.asarray([1,3,4])
    w, grads_norms = SGD(X, w,y,  200)
    print(w)
    plot_graphs(grads_norms)

def plot_graphs(grads_norms):
    plt.semilogy(grads_norms, label='grads_norms' )
    plt.suptitle("SGD")
    plt.xlabel("iteration")
    plt.ylabel('f_values')
    plt.legend()
    plt.show()

test_SGD_LS_2()