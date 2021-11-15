import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

GMM = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\GMMData.mat')
Peaks = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\PeaksData.mat')
SwissRoll = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\SwissRollData.mat')

Ct = SwissRoll["Ct"]
Cv = SwissRoll["Cv"]
yt = SwissRoll["Yt"]
yv = SwissRoll["Yv"]



def soft_max_regression(X,C, W):
    m = len(X[0])
    eta = calculate_eta_vector(X, W)
    f = 0
    for i in range(len(W[0])):
        f += C[i] @ (np.log(np.exp((X.transpose() @ W[:,i]) - eta) / calculate_divider(X, W)))
    return (-1/m) * f



def calculate_divider(X,W):
    sum = np.zeros(len(X[0]))
    eta = calculate_eta_vector(X, W)
    for i in range(len(W[0])):
        sum += np.exp((X.transpose() @ W[:,i]) - eta)
    return sum


# def calculate_divider(X,W):
#     return np.ndarray.sum(np.exp((X.transpose() @ W) - calculate_neu(X,W)), axis=1)

# def calculate_neu(X, W):
#     m = len(X[0])
#     neu = np.zeros(m)
#     for i in range(m):
#         for j in range(len(W)):
#             neu[i] = max(neu[i], X[:,i] @ W[:,j])
#     return np.repeat(neu[:,np.newaxis], len(W[0]), 1)

def calculate_eta_vector(X, W):
    return np.max(X.transpose() @ W)

# def grad_soft_max(X,W,C):
#     gradW = []
#     for i in range(len(W[0])):
#         Wp = (1/len(X[0])) * (X @ ((np.exp((X.transpose() @ W[:,i]) - calculate_eta_vector(X, W)) / calculate_divider(X, W)) - C[:, i]))
#         gradW.append(Wp)
#     return np.asarray(gradW).transpose()

def grad_soft_max(X,W,C):
    gradW = []
    for i in range(len(W[0])):
        Wp = (1/len(X[0])) * (X @ ((np.exp((X.T @ W[:,i]) - calculate_eta_vector(X, W)) / calculate_divider(X, W)) - C[i]))
        gradW.append(Wp)
    return np.asarray(gradW).transpose()


def grad_test():
    X = yt
    C = Ct
    # X = np.random.rand(10, 20)
    # C = np.random.rand(20, 15)
    W = np.random.rand(2,2)
    D = np.random.rand(2,2)
    D = (1/LA.norm(D)) * D
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result = soft_max_regression(X, C, W)
    grad = grad_soft_max(X, W, C)
    for i in range(20):
        func_with_epsilon = soft_max_regression(X,C,W + (epsilon*D))
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon-func_result-(epsilon*(np.ndarray.flatten(D,'F') @ np.ndarray.flatten(grad,'F')))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="soft max loss")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="grad loss")
    plt.xlabel('epsilons')
    plt.ylabel('Decrease factor')
    plt.legend()
    plt.show()
    print("soft max values: ", soft_max_loss)
    print("grad values: ", grad_soft_max_loss)

grad_test()


# -------------------------------------------------------task 2.1.3------------------------------------------------

def SGD(obj, grad, X, w, c, epoch):
    norms = [LA.norm(c)]
    lr = 0.1
    batch = 1
    obj_values = [obj(X,w,c)]
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
        norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
        obj_values.append(obj(X,w,c))
    return w, norms, obj_values


def pick_sample(X,c, m):
    perm = np.random.permutation(len(X))
    indx = perm[0:m]
    sampleX = X[indx, :]
    samplec = c[indx]
    return sampleX, samplec

def test_data():
    w, norms, obj_values = SGD(soft_max_regression())