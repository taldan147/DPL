import numpy as np
import scipy.io as sio
import numpy.linalg as LA
import matplotlib.pyplot as plt

GMM = sio.loadmat('../GMMData.mat')
Peaks = sio.loadmat('../PeaksData.mat')
SwissRoll = sio.loadmat('../SwissRollData.mat')

Ct = Peaks["Ct"]
Cv = Peaks["Cv"]
yt = Peaks["Yt"]
yv = Peaks["Yv"]

def soft_max(X,W):
    eta = calculate_eta_vector(X,W)
    return np.exp((X.T @ W).T - eta) / np.sum(np.exp((X.T @ W).T - eta).T, axis=1)

def soft_max_regression(X,C, W):
    m = len(X[0])
    f = np.sum(C * np.log(soft_max(X,W)))
    return (-1/m) * f

def grad_soft_max(X,W,C):
    m = len(X[0])
    grad = X @ (soft_max(X,W).T - C.T)
    return 1/m * grad

def calculate_eta_vector(X, W):
    return np.max(X.transpose() @ W, axis=1)

def grad_test():
    X = yt
    X = np.vstack([X, np.ones(len(X[0]))])  # adding ones for the bias
    C = Ct
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
    plt.title('softmax regression grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

# grad_test()