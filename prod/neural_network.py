import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import scipy.io as sio
from prod.task1_1 import soft_max, soft_max_regression, grad_soft_max, calculate_eta_vector
from prod.task1_3 import Cv_Peaks, Ct_Peaks, Ct_GMM , Cv_GMM, Ct_SwissRoll, Cv_SwissRoll, yv_Peaks, yt_Peaks, yt_SwissRoll, yt_GMM, yv_GMM, yv_SwissRoll, classify

def forward_pass(f, X, W, l, C):
    keeper_X = [X]
    X_i = X
    for i in range(l - 1):
        X_i = f((W[i] @ X_i))
        keeper_X.append(X_i)
    return soft_max_regression(keeper_X[l-1], C, W[l-1]), keeper_X, soft_max(keeper_X[l-1], W[l-1])

def derive_by_X(X, W, v):
    return W.T @ derive_by_b(X, W, v)

def derive_by_b(X, W, v):
    return (tanh_derivative((W @ X)) * v)

def derive_by_W(X, W, v):
    by_b = derive_by_b(X, W, v)
    return by_b @ X.T

def grad_soft_max_by_X(X,W,C):
    eta = calculate_eta_vector(X,W)
    return (1/len(X[0])) * (W @ ((np.exp(W.T@X - eta) / np.sum(np.exp(W.T @ X - eta), axis=0)) - C))

def back_propagation(keeper_X, W, l, C):
    grad = [grad_soft_max(keeper_X[l-1], W[l-1], C)]
    deriv_by_x = grad_soft_max_by_X(keeper_X[l-1],W[l-1],C)
    for i in range(l-2, -1, -1):
        dw = derive_by_W(keeper_X[i], W[i], deriv_by_x)
        grad.append(dw)
        deriv_by_x = derive_by_X(keeper_X[i], W[i], deriv_by_x)
    return grad

def tanh_derivative(X):
    return np.ones(np.shape(X)) - np.power(np.tanh(X), 2)

def test_jacobian_W():
    X = np.append(yt_Peaks[:,0], 1).reshape(3,1)
    u = np.random.rand(3,1)
    W = np.random.rand(3,3)
    D_W = np.random.rand(3,3)
    D_W = (1 / LA.norm(D_W)) * D_W
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = (np.tanh(W @ X).T @ u).flatten()
    dw = derive_by_W(X, W, u)
    for i in range(20):
        func_with_epsilon = (np.tanh((W +epsilon*D_W) @ X).T @ u).flatten()
        f_loss.append(abs(func_with_epsilon - func_result))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon * (np.ndarray.flatten(D_W) @ np.ndarray.flatten(dw)))))
        epsilon *= 0.5
    plt.figure()
    plt.semilogy([i for i in range(20)], f_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad__loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.suptitle('Jacobian transpose test for the derivative of the layer by W')
    plt.legend()
    plt.show()

def test_jacobian_X():
    X = np.append(yt_Peaks[:,0], 1).reshape(3,1)
    u = np.random.rand(3,1)
    W = np.random.rand(3,3)
    D_X = np.random.rand(3,1)
    D_X = (1 / LA.norm(D_X)) * D_X
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = (np.tanh(W @ X).T @ u).flatten()
    dx = derive_by_X(X, W, u)
    for i in range(20):
        func_with_epsilon = (np.tanh(W @ (X + epsilon*D_X)).T @ u).flatten()
        f_loss.append(abs(func_with_epsilon - func_result))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon * (np.ndarray.flatten(D_X) @ np.ndarray.flatten(dx)))))
        epsilon *= 0.5
    plt.figure()
    plt.semilogy([i for i in range(20)], f_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad__loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.suptitle('Jacobian transpose test for the derivative of the layer by X')
    plt.legend()
    plt.show()

def test_grad_whole_network():
    X = yt_Peaks
    X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct_Peaks
    W = [np.random.rand(4,3),np.random.rand(4,4), np.random.rand(4,5)]
    b = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    d_W = [np.random.rand(4,3),np.random.rand(4,4), np.random.rand(4,5)]
    d_B = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result, keeper_X, no = forward_pass(np.tanh, X, W, len(W), C)
    grad = back_propagation(keeper_X, W, len(W), C)
    flat_d = np.asarray([*(d_W[2]).flatten(), *(d_W[1]).flatten(), *(d_W[0]).flatten()])
    flat_grad = np.asarray([*(grad[0]).flatten(), *(grad[1]).flatten(), *(grad[2]).flatten()])
    for i in range(20):
        new_W = [(W[0]+epsilon*d_W[0]), (W[1]+epsilon*d_W[1]), (W[2]+epsilon*d_W[2])]
        new_B = [b[0] + epsilon * d_B[0], b[1] + epsilon * d_B[1], b[2] + epsilon * d_B[2]]
        func_with_epsilon, notIntresting, no2 = forward_pass(np.tanh, X, new_W, len(W), C)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon - func_result - (epsilon * (flat_d @ flat_grad))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Gradient test for the Neural Network')
    plt.legend()
    plt.show()

def SGD_NN(back_prop, X, w, c,X_valid, C_valid, epoch, batch):
    lr = 0.5
    success_percentages = [calculate_success_NN(X,w,c)]
    success_percentages_valid = [calculate_success_NN(X_valid,w,C_valid)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        # lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.5
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currC = c[:, indx]
            no1, keeperX, no2 = forward_pass(np.tanh, currX, w, len(w), currC)
            gradK = back_prop(keeperX, w, len(w), currC)
            w = update_param(w, gradK, lr)
        success_percentages.append(calculate_success_NN(X,w,c))
        success_percentages_valid.append(calculate_success_NN(X_valid,w,C_valid))
    return w, success_percentages, success_percentages_valid

def update_param(W, grad, lr):
    l = len(grad)
    for i in range(l-1, -1, -1):
        W[l-i-1] -= lr*grad[i]
    return W


def calculate_success_NN(X, W, C):
    no1, no2, SMresult = forward_pass(np.tanh, X, W, len(W), C)
    classified = classify(X, W[len(W)-1], SMresult)
    return 1 - np.sum(abs(C - classified)) / (2 * len(X[0]))


def test_NN(X_train, X_valid, C_train, C_valid, W,  epoch, batch, data):
    X_train = np.vstack([X_train, np.ones(len(X_train[0]))])
    X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    w_train, success_percentages_train, success_percentages_validation = SGD_NN(back_propagation, X_train, W, C_train,X_valid, C_valid, epoch, batch)
    plt.plot(np.arange(len(success_percentages_train)), [x*100 for x in success_percentages_train], label='train')
    plt.plot(np.arange(len(success_percentages_validation)), [x*100 for x in success_percentages_validation], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    # plt.title('Success percentages of NN for %s,\n' % data + r' lr: $\frac{1}{\sqrt{epoch}}$, batch: %s' % (batch))
    plt.title('Success percentages of NN for ' + data + ',\n lr: 0.5, batch: ' + str(batch))
    print("avg train success percentage of ", data, ":",
          success_percentages_train[99])
    print("avg validation success percentage of ", data, ":",
          success_percentages_validation[99])
    # print("avg train success percentage of ", data, ":",
    #       np.mean(np.delete(np.asarray(success_percentages_train), np.arange(5))))
    # print("avg validation success percentage of ", data, ":",
    #       np.mean(np.delete(np.asarray(success_percentages_validation), np.arange(5))))
    plt.legend()
    plt.show()

def test_NN_peaks():
    W = [np.random.rand(10, 3),np.random.rand(10, 10),np.random.rand(4, 10), np.random.rand(4, 5)]
    epoch = 100
    batch = 100
    test_NN(yt_Peaks, yv_Peaks, Ct_Peaks, Cv_Peaks, W, epoch, batch, "Peaks")

def test_NN_peaks_200():
    W = [np.random.rand(10, 3),np.random.rand(10, 10),np.random.rand(4, 10), np.random.rand(4, 5)]
    epoch = 400
    batch = 100
    X,C = pick_sample(yt_Peaks, Ct_Peaks, 200)
    test_NN(X, yv_Peaks, C,Cv_Peaks, W,  epoch, batch, "Peaks")


def test_NN_SwissRoll():
    W = [np.random.rand(10, 3), np.random.rand(10, 10), np.random.rand(4, 10), np.random.rand(4, 2)]
    epoch = 100
    batch = 100
    test_NN(yt_SwissRoll, yv_SwissRoll, Ct_SwissRoll, Cv_SwissRoll, W, epoch, batch, "SwissRoll")

def test_NN_SwissRoll_200():
    W = [np.random.rand(10, 3), np.random.rand(10, 10), np.random.rand(4, 10), np.random.rand(4, 2)]
    epoch = 100
    batch = 100
    X, C = pick_sample(yt_SwissRoll, Ct_SwissRoll, 200)
    test_NN(X, yv_SwissRoll, C, Cv_SwissRoll, W, epoch, batch, "SwissRoll")

def test_NN_GMM():
    W = [np.random.rand(10, 6), np.random.rand(4, 10), np.random.rand(4, 5)]
    epoch = 100
    batch = 100
    test_NN(yt_GMM, yv_GMM, Ct_GMM, Cv_GMM, W, epoch, batch, "GMM")

def pick_sample(X,c, m):
    perm = np.random.permutation(len(X[0]))
    indx = perm[0: m]
    sampleX = X[:, indx]
    samplec = c[:, indx]
    return sampleX, samplec

# test_NN_peaks_200()
# test_NN_peaks()
# test_NN_SwissRoll()
# test_NN_GMM()


# test_jacobian_W()
# test_jacobian_X()
# test_grad_whole_network()
