import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

GMM = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\GMMData.mat')
Peaks = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\PeaksData.mat')
SwissRoll = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\SwissRollData.mat')

Ct = Peaks["Ct"]
Cv = Peaks["Cv"]
yt = Peaks["Yt"]
yv = Peaks["Yv"]



def soft_max(X,W):
    eta = calculate_eta_vector(X,W)
    return np.exp((X.T @ W).T - eta) / np.sum(np.exp((X.T @ W).T - eta).T, axis=1)

def soft_max_regression(X,C, W):
    m = len(X[0])
    # eta = calculate_eta_vector(X, W)
    # f = 0
    # for i in range(len(W[0])):
    #     f += C[i] @ (np.log(np.exp((X.transpose() @ W[:,i]) - eta) / calculate_divider(X, W)))
    # return (-1/m) * f
    f = np.sum(C * np.log(soft_max(X,W)))
    return (-1/m) * f

def grad_soft_max(X,W,C):
    m = len(X[0])
    grad = X @ (soft_max(X,W).T - C.T)
    return 1/m * grad
    # gradW = []
    # for i in range(len(W[0])):
    #     Wp = (1/len(X[0])) * (X @ ((np.exp((X.T @ W[:,i]) - calculate_eta_vector(X, W)) / calculate_divider(X, W)) - C[i]))
    #     gradW.append(Wp)
    # return np.asarray(gradW).transpose()

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
    return np.max(X.transpose() @ W, axis=1)

# def grad_soft_max(X,W,C):
#     gradW = []
#     for i in range(len(W[0])):
#         Wp = (1/len(X[0])) * (X @ ((np.exp((X.transpose() @ W[:,i]) - calculate_eta_vector(X, W)) / calculate_divider(X, W)) - C[:, i]))
#         gradW.append(Wp)
#     return np.asarray(gradW).transpose()




def grad_test():
    X = yt
    C = Ct
    # X = np.random.rand(10, 20)
    # C = np.random.rand(20, 15)
    n = len(X)
    l = len(C)
    W = np.random.rand(n, l)
    D = np.random.rand(n, l)
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
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

grad_test()


# -------------------------------------------------------task 2.1.3------------------------------------------------

def SGD(grad, X, w, c, epoch):
    # norms = []
    # lr = 0.1
    batch = 500
    success_percentages = [calculate_success(X,w,c)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currc = c[:, indx]
            gradK = grad(currX, w, currc) + 0.01*w
            w = w-lr*gradK
        # norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
        success_percentages.append(calculate_success(X,w,c))
    return w, success_percentages


def pick_sample(X,c, m):
    perm = np.random.permutation(len(X))
    indx = perm[0:m]
    sampleX = X[indx, :]
    samplec = c[indx]
    return sampleX, samplec

def classify(X,W):
    m = len(X[0])
    l = len(W[0])
    labels = np.argmax(soft_max(X,W), axis=0)
    classified_matrix = np.zeros((l,m))
    classified_matrix[labels, np.arange(m)] = 1
    return classified_matrix

def calculate_success(X,W,C):
    return 1 - np.sum(abs(C - classify(X,W))) / (2*len(X[0]))


def test_data():
    X = yt
    C = Ct
    X_valid = yv
    C_valid = Cv
    W = np.random.rand(len(X), len(C))
    w_train, success_percentages_train = SGD(grad_soft_max, X, W, C, 300)
    w_train, success_percentages_validation = SGD(grad_soft_max, X_valid, W, C_valid, 300)
    plt.plot(np.arange(len(success_percentages_train)), success_percentages_train, label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), success_percentages_validation, label='success percentage for validation per epoch')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

# test_data()

