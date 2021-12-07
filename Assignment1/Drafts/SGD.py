import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

def SGD(grad, X, w, c, epoch):
    norms = [LA.norm(c)]
    lr = 0.01
    batch = 1
    for i in range(epoch):
        perm = np.random.permutation(len(X))
        # lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[indx, :]
            currc = c[indx]
            gradK = grad(currX, w, currc) + 0.01*w
            w = w-lr*gradK
        norms.append(LA.norm(grad(X,w,c) + 0.01*w))
    return w, norms

def test_SGD_LS_2():
    X = np.asarray([[-1,-1,1], [1,3,3],[-1,-1,5],[1,3,7]])
    c = np.asarray([0,23,15,39])
    w = np.zeros(3)
    # w = np.asarray([1,3,4])
    w, grads_norms = SGD(LS_grad, X, w,c,  200)
    print(w)
    plot_graphs(grads_norms)

def LS_grad(X,w,c):
    grad = np.zeros(len(w))
    for i in range(len(X)):
        grad += ((X[i] @ w) -c[i]) * X[i]
    return 1/len(X) * grad

def plot_graphs(grads_norms):
    plt.semilogy(grads_norms, label='grads_norms')
    plt.suptitle("minimizing LS with SGD")
    plt.xlabel("epoch")
    plt.ylabel('gradient norm')
    plt.legend()
    plt.show()

test_SGD_LS_2()