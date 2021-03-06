import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

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

def grad_soft_max_b(X,W,C):
    m = len(X[0])
    return 1/m * soft_max(X,W).T - C.T

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
    X = np.vstack([X, np.ones(len(X[0]))])
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

# grad_test()


# -------------------------------------------------------task 2.1.3------------------------------------------------

def SGD(grad, X, w, c, epoch, batch):
    # norms = []
    lr = 1
    # batch = 6000
    # success_percentages = [soft_max_regression(X,c,w)]
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
        # success_percentages.append(soft_max_regression(X,c,w))
        success_percentages.append(calculate_success(X,w,c))
    return w, success_percentages


def pick_sample(X,c, m):
    perm = np.random.permutation(len(X))
    indx = perm[0:m]
    sampleX = X[indx, :]
    samplec = c[indx]
    return sampleX, samplec

def classify(X,W, probs_matrix):
    m = len(X[0])
    l = len(W[0])
    labels = np.argmax(probs_matrix, axis=0)
    classified_matrix = np.zeros((l,m))
    classified_matrix[labels, np.arange(m)] = 1
    return classified_matrix

def calculate_success(X,W,C):
    # X = np.delete(X, len(X)-1, 0)
    # classified = np.delete(classify(X,W), len(X)-1, 0)
    return 1 - np.sum(abs(C - classify(X,W, soft_max(X, W)))) / (2*len(X[0]))


def test_data():
    X = yt
    X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    X_valid = yv
    X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    C_valid = Cv
    W = np.random.rand(len(X), len(C))
    epoch =200
    w_train, success_percentages_train = SGD(grad_soft_max, X, W, C, epoch, 10000)
    w_train_valid, success_percentages_validation = SGD(grad_soft_max, X_valid, W, C_valid, epoch, 1000)
    plt.plot(np.arange(len(success_percentages_train)), [x*100 for x in success_percentages_train], label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), [x*100 for x in success_percentages_validation], label='success percentage for validation per epoch')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

# test_data()


# ---------------------------------------------------------------------------task2.2-------------------------------------------------------------------------

def forward_pass(f, X, W, B, l, C):
    keeper_X = [X]
    X_i = X
    for i in range(l - 1):
        # X_i = f((W[i] @ X_i) + B[i].reshape(len(B[i]), 1))
        X_i = f((W[i] @ X_i))
        keeper_X.append(X_i)
    # keeper_X[l-1] = np.vstack([X_i, np.ones(len(X_i[0]))])
    return soft_max_regression(keeper_X[l-1], C, W[l-1]), keeper_X, soft_max(keeper_X[l-1], W[l-1])

def derive_by_X(X, W, b, v):
    # X = np.delete(X, len(X) - 1, 0)
    # W = np.delete(W, len(W) - 1, 0)
    return W.T @ derive_by_b(X, W, b, v)

def derive_by_b(X, W, b, v):
    return (tanh_derivative((W @ X)) * v)
    # return (tanh_derivative((W @ X).T + b).T * v)

def derive_by_W(X, W, b, v):
    by_b = derive_by_b(X, W, b, v)
    return by_b @ X.T
    # return np.reshape(by_b, (len(by_b), 1)) @ np.reshape(X, (1,len(X)))

def grad_soft_max_by_X(X,W,C):
    eta = calculate_eta_vector(X,W)
    # X = np.delete(X, len(X)-1, 0)
    # W = np.delete(W, len(W)-1, 0)
    return (1/len(X[0])) * (W @ ((np.exp(W.T@X - eta) / np.sum(np.exp(W.T @ X - eta), axis=0)) - C))

def back_propagation(keeper_X, W, B, l, C):
    grad = [grad_soft_max(keeper_X[l-1], W[l-1], C)]
    deriv_by_x = grad_soft_max_by_X(keeper_X[l-1],W[l-1],C)
    for i in range(l-2, -1, -1):
        dw = derive_by_W(keeper_X[i], W[i], 0, deriv_by_x)
        # db = np.sum(derive_by_b(keeper_X[i], W[i], B[i], deriv_by_x), axis=1)
        # curr_deriv_by_theta = np.append(dw, np.reshape(db, (len(db),1)), axis=1)
        # grad.append(curr_deriv_by_theta)
        grad.append(dw)
        deriv_by_x = derive_by_X(keeper_X[i], W[i], 0, deriv_by_x)
    return grad

def tanh_derivative(X):
    return np.ones(np.shape(X)) - np.power(np.tanh(X), 2)

def test_jacobian():
    X = yt[:,0]
    u = np.random.rand(2)
    b = np.random.rand(2)
    W = np.random.rand(2,5)
    D_b = np.random.rand(2)
    D_b = (1 / LA.norm(D_b)) * D_b
    D_W = np.random.rand(2,5)
    D_W = (1 / LA.norm(D_W)) * D_W
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = np.tanh((W @ X) + b) @ u
    dw = derive_by_W(X, W, b, u)
    db = derive_by_b(X, W, b, u)
    grad = np.append(dw, np.reshape(db, (2,1)), axis=1)
    D = np.append(D_W, np.reshape(D_b, (2,1)), axis=1)
    for i in range(20):
        # func_with_epsilon = np.tanh(W @ X + (b+D_b*epsilon)) @ u
        func_with_epsilon = np.tanh((W +epsilon*D_W) @ X + (b+epsilon*D_b)) @ u
        f_loss.append(abs(func_with_epsilon - func_result))
        # grad__loss.append(abs(
        #     func_with_epsilon - func_result - (epsilon * (D_b @ db))))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon * (np.ndarray.flatten(D) @ np.ndarray.flatten(grad)))))
        epsilon *= 0.5
    plt.figure()
    plt.semilogy([i for i in range(20)], f_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad__loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

# test_jacobian()


def test_grad_whole_network():
    # X = yt[:,0].reshape(2, 1)
    X = yt
    X = np.vstack([X, np.ones(len(X[0]))])
    # C = Ct[:,0].reshape(1, 5)
    C = Ct
    W = [np.random.rand(4,3),np.random.rand(4,4), np.random.rand(4,5)]
    b = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    d_W = [np.random.rand(4,3),np.random.rand(4,4), np.random.rand(4,5)]
    d_B = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result, keeper_X, no = forward_pass(np.tanh, X, W, b, len(W), C)
    grad = back_propagation(keeper_X, W,b, len(W), C)
    flat_d = np.asarray([*(d_W[2]).flatten(), *(d_W[1]).flatten(), *(d_W[0]).flatten()])
    # flat_d = np.asarray([*(d_W[2]).flatten(), *(np.append(d_W[1], d_B[1].reshape(3, 1), axis=1)).flatten(), *(np.append(d_W[0], d_B[0].reshape(3, 1), axis=1)).flatten()])
    flat_grad = np.asarray([*(grad[0]).flatten(), *(grad[1]).flatten(), *(grad[2]).flatten()])
    for i in range(20):
        new_W = [(W[0]+epsilon*d_W[0]), (W[1]+epsilon*d_W[1]), (W[2]+epsilon*d_W[2])]
        new_B = [b[0] + epsilon * d_B[0], b[1] + epsilon * d_B[1], b[2] + epsilon * d_B[2]]
        func_with_epsilon, notIntresting, no2 = forward_pass(np.tanh, X, new_W, b, len(W), C)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon - func_result - (epsilon * (flat_d @ flat_grad))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

# test_grad_whole_network()

def grad_test_soft_max_by_X():
    X = yt
    # X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    # X = np.random.rand(10, 20)
    # C = np.random.rand(20, 15)
    n = len(X)
    l = len(C)
    W = np.random.rand(n, l)
    D = np.random.rand(n, len(X[0]))
    D = (1/LA.norm(D)) * D
    D_deleted = np.delete(D, len(D)-1, 0)
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result = soft_max_regression(X, C, W)
    grad = grad_soft_max_by_X(X, W, C)
    for i in range(20):
        func_with_epsilon = soft_max_regression(X+ (epsilon*D),C,W)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon-func_result-(epsilon*(np.ndarray.flatten(D) @ np.ndarray.flatten(grad)))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

# grad_test_soft_max_by_X()

def SGD_NN(back_prop, X, w,b, c, epoch, batch):
    # norms = []
    lr = 1
    # batch = 6000
    # success_percentages = [soft_max_regression(X,c,w)]
    success_percentages = [calculate_success_NN(X,w,c)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currC = c[:, indx]
            no1, keeperX, no2 = forward_pass(np.tanh, currX, w, b, len(w), currC)
            gradK = back_prop(keeperX, w, b, len(w), currC)
            w = update_param(w, gradK, lr)
        # norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
        # success_percentages.append(soft_max_regression(X,c,w))
        success_percentages.append(calculate_success_NN(X,w,c))
    return w, success_percentages

def update_param(W, grad, lr):
    l = len(grad)
    # for i in range(l-1, -1, -1):
        W[l-i-1] -= lr*grad[i]
    return W

def calculate_success_NN(X, W, C):
    no1, no2, SMresult = forward_pass(np.tanh, X, W, [], len(W), C)
    classified = classify(X, W[len(W)-1], SMresult)
    return 1 - np.sum(abs(C - classified)) / (2 * len(X[0]))

def test_NN():
    X = yt
    X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    W = [np.random.rand(10,3),np.random.rand(4,10), np.random.rand(4,5)]
    W_valid = W.copy()
    X_valid = yv
    X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    C_valid = Cv
    epoch = 1500
    w_train, success_percentages_train = SGD_NN(back_propagation, X, W, 0, C, epoch, 1000)
    w_train_valid, success_percentages_validation = SGD_NN(back_propagation, X_valid, W_valid, 0, C_valid, epoch, 1000)
    plt.plot(np.arange(len(success_percentages_train)), [x*100 for x in success_percentages_train], label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), [x*100 for x in success_percentages_validation], label='success percentage for validation per epoch')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

test_NN()