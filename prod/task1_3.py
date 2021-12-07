import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from prod.task1_1 import soft_max, soft_max_regression, grad_soft_max

GMM = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\GMMData.mat')
Peaks = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\PeaksData.mat')
SwissRoll = sio.loadmat('C:\\Users\\tald9\\PycharmProjects\\DPL\\task2\\SwissRollData.mat')

Ct_Peaks = Peaks["Ct"]
Cv_Peaks = Peaks["Cv"]
yt_Peaks = Peaks["Yt"]
yv_Peaks = Peaks["Yv"]

Ct_GMM = GMM["Ct"]
Cv_GMM = GMM["Cv"]
yt_GMM = GMM["Yt"]
yv_GMM = GMM["Yv"]

Ct_SwissRoll = SwissRoll["Ct"]
Cv_SwissRoll = SwissRoll["Cv"]
yt_SwissRoll = SwissRoll["Yt"]
yv_SwissRoll = SwissRoll["Yv"]


def SGD(grad, X, w, c,X_valid, c_valid, epoch, batch):
    lr = 0.1
    success_percentages = [calculate_success(X,w,c)]
    success_percentages_valid = [calculate_success(X_valid,w,c_valid)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        # lr = 1/(math.sqrt(1+i))
        if i % 50 == 0:
            lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currc = c[:, indx]
            gradK = grad(currX, w, currc) + 0.01*w
            w = w-lr*gradK
        success_percentages.append(calculate_success(X,w,c))
        success_percentages_valid.append(calculate_success(X_valid,w,c_valid))
    return w, success_percentages, success_percentages_valid

def classify(X,W, probs_matrix):
    m = len(X[0])
    l = len(W[0])
    labels = np.argmax(probs_matrix, axis=0)
    classified_matrix = np.zeros((l,m))
    classified_matrix[labels, np.arange(m)] = 1
    return classified_matrix

def calculate_success(X,W,C):
    return 1 - np.sum(abs(C - classify(X,W, soft_max(X, W)))) / (2*len(X[0]))

def test_data(X, X_valid, C, C_valid, data):
    X = np.vstack([X, np.ones(len(X[0]))])
    X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    W = np.random.rand(len(X), len(C))
    epoch = 100
    w_train, success_percentages_train, success_percentages_validation = SGD(grad_soft_max, X, W, C,X_valid, C_valid, epoch, 20)
    plt.plot(np.arange(len(success_percentages_train)), [x*100 for x in success_percentages_train], label='train')
    plt.plot(np.arange(len(success_percentages_validation)), [x*100 for x in success_percentages_validation], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.title("success percentage of minimize " + data + " with SGD,\n lr:0.1, batch:200")
    # plt.title("success percentage of minimize " + data + r" with SGD, lr:$\frac{1}{\sqrt{epoch}}$, batch:200")
    plt.legend()
    plt.show()
    print("avg train success percentage of ", data, ":",np.mean(np.delete(np.asarray(success_percentages_train), np.arange(5))), ", ",success_percentages_train[99])
    print("avg validation success percentage of ", data, ":",np.mean(np.delete(np.asarray(success_percentages_validation), np.arange(5))), ", ",success_percentages_train[99])

# test_data(yt_Peaks,yv_Peaks, Ct_Peaks, Cv_Peaks, "Peaks")
# test_data(yt_GMM,yv_GMM, Ct_GMM, Cv_GMM, "GMM")
# test_data(yt_SwissRoll,yv_SwissRoll, Ct_SwissRoll, Cv_SwissRoll, "SwissRoll")

