import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import random
import scipy.io as sio

GMM = sio.loadmat('GMMData.mat')
Peaks = sio.loadmat('PeaksData.mat')
SwissRoll = sio.loadmat('SwissRollData.mat')

Ct = Peaks["Ct"]
Cv = Peaks["Cv"]
yt = Peaks["Yt"]
yv = Peaks["Yv"]


def soft_max(X, W, b):
    eta = calculate_eta_vector(X, W, b)
    return (np.exp(((X.T @ W).T - eta).T + b).T / np.sum(np.exp(((X.T @ W).T - eta).T + b), axis=1))


def soft_max_regression(X, C, W, b):
    m = len(X[0])
    f = np.sum(C.T * np.log(soft_max(X, W, b)))
    return (-1 / m) * f


def grad_soft_max(X, W, C, b):
    m = len(X[0])
    grad = X @ (soft_max(X, W, b).T - C)
    return 1 / m * grad


def grad_soft_max_b(X, W, C, b):
    m = len(X[0])
    return 1 / m * np.sum(soft_max(X, W, b) - C, axis=1)


# def calculate_divider(X,W):
#     return np.ndarray.sum(np.exp((X.transpose() @ W) - calculate_neu(X,W)), axis=1)

# def calculate_neu(X, W):
#     m = len(X[0])
#     neu = np.zeros(m)
#     for i in range(m):
#         for j in range(len(W)):
#             neu[i] = max(neu[i], X[:,i] @ W[:,j])
#     return np.repeat(neu[:,np.newaxis], len(W[0]), 1)

def calculate_eta_vector(X, W, b):
    return np.max(X.transpose() @ W + b, axis=1)


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
    b = np.random.rand(l)
    # D = np.random.rand(n, l)
    D = np.random.rand(l)
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result = soft_max_regression(X, C, W, b)
    grad = grad_soft_max_b(X, W, C, b)
    # grad = grad_soft_max(X, W, C, b)
    for i in range(20):
        func_with_epsilon = soft_max_regression(X, C, W, b + (epsilon * D))
        # func_with_epsilon = soft_max_regression(X,C,W + (epsilon*D), b)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon - func_result - (epsilon * (D @ grad))))
        # grad_soft_max_loss.append(abs(func_with_epsilon-func_result-(epsilon*(np.ndarray.flatten(D,'F') @ np.ndarray.flatten(grad,'F')))))
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

def SGD(grad, X, w, c, epoch, b):
    # norms = []
    lr = 1
    batch = 2000
    # success_percentages = [soft_max_regression(X,c,w)]
    success_percentages = [calculate_success(X, w, c, b)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        lr = 1 / (math.sqrt(1 + i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0]) / batch)):
            indx = perm[k * batch:(k + 1) * batch]
            currX = X[:, indx]
            currc = c[:, indx]
            gradK = grad(currX, w, currc) + 0.01 * w
            w = w - lr * gradK
        # norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
        # success_percentages.append(soft_max_regression(X,c,w))
        success_percentages.append(calculate_success(X, w, c, b))
    return w, success_percentages


def pick_sample(X, c, m):
    perm = np.random.permutation(len(X))
    indx = perm[0:m]
    sampleX = X[indx, :]
    samplec = c[indx]
    return sampleX, samplec


def classify(X, W, b):
    m = len(X[0])
    l = len(W[0])
    labels = np.argmax(soft_max(X, W, b), axis=0)
    classified_matrix = np.zeros((l, m))
    classified_matrix[labels, np.arange(m)] = 1
    return classified_matrix


def calculate_success(X, W, C, b):
    return 1 - np.sum(abs(C - classify(X, W, b))) / (2 * len(X[0]))


def test_data():
    X = yt
    C = Ct
    X_valid = yv
    C_valid = Cv
    W = np.random.rand(len(X), len(C))
    b = np.random.rand(len(C))
    epoch = 200
    w_train, success_percentages_train = SGD(grad_soft_max, X, W, C, epoch, b)
    w_train_valid, success_percentages_validation = SGD(grad_soft_max, X_valid, W, C_valid, epoch, b)
    plt.plot(np.arange(len(success_percentages_train)), [x * 100 for x in success_percentages_train],
             label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), [x * 100 for x in success_percentages_validation],
             label='success percentage for validation per epoch')
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
        X_i = f((W[i] @ X_i) + B[i].reshape(len(B[i]), 1))
        keeper_X.append(X_i)
    return soft_max_regression(X_i, C, W[l - 1], B[l - 1]), keeper_X  # ????????????????????????


def derive_by_X(X, W, b, v):
    return W.T @ derive_by_b(X, W, b, v)


def derive_by_b(X, W, b, v):
    return (tanh_derivative((W @ X).T + b) * v.T).T


def derive_by_W(X, W, b, v):
    by_b = derive_by_b(X, W, b, v)
    return by_b @ X.T


def grad_soft_max_by_X(X, W, C):
    return (1 / len(X[0])) * W @ (np.exp(W.T @ X) / np.sum(W.T @ X, axis=0) - C.T)


def back_propagation(keeper_X, W, B, l, C):
    # grad = [grad_soft_max_b(keeper_X[l-1], W[l-1], C, B[l-1])]
    grad = [np.append(grad_soft_max(keeper_X[l - 1], W[l - 1], C, B[l - 1]),
                      grad_soft_max_b(keeper_X[l - 1], W[l - 1], C, B[l - 1]), axis=1)]
    deriv_by_x = grad_soft_max_by_X(keeper_X[l - 1], W[l - 1], C)
    for i in range(l - 2, -1, -1):
        dw = derive_by_W(keeper_X[i], W[i], B[i], deriv_by_x)
        db = derive_by_b(keeper_X[i], W[i], B[i], deriv_by_x)
        curr_deriv_by_theta = np.append(dw, db, axis=1)
        grad.append(curr_deriv_by_theta)
        deriv_by_x = derive_by_X(keeper_X[i], W[i], B[i], deriv_by_x)
    return (grad)


def tanh_derivative(X):
    return np.ones(np.shape(X)) - np.power(np.tanh(X), 2)


def test_jacobian():
    X = yt[:, 0]
    u = np.random.rand(2)
    b = np.random.rand(2)
    W = np.random.rand(2, 5)
    D_b = np.random.rand(2)
    D_b = (1 / LA.norm(D_b)) * D_b
    D_W = np.random.rand(2, 5)
    D_W = (1 / LA.norm(D_W)) * D_W
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = np.tanh((W @ X) + b) @ u
    dw = derive_by_W(X, W, b, u)
    db = derive_by_b(X, W, b, u)
    grad = np.append(dw, np.reshape(db, (2, 1)), axis=1)
    D = np.append(D_W, np.reshape(D_b, (2, 1)), axis=1)
    for i in range(20):
        # func_with_epsilon = np.tanh(W @ X + (b+D_b*epsilon)) @ u
        func_with_epsilon = np.tanh((W + epsilon * D_W) @ X + (b + epsilon * D_b)) @ u
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
    X = yt[:, 0].reshape(2, 1)
    C = Ct[:, 0].reshape(1, 5)
    W = [np.random.rand(3, 2), np.random.rand(3, 3), np.random.rand(3, 5)]
    b = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    d_W = [np.random.rand(3, 2), np.random.rand(3, 3), np.random.rand(3, 5)]
    d_B = [np.random.rand(3), np.random.rand(3), np.random.rand(5)]
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result, keeper_X = forward_pass(np.tanh, X, W, b, len(W), C)
    grad = back_propagation(keeper_X, W, b, len(W), C)
    flat_d = np.asarray([*d_W[2].T, *np.append(d_W[1], d_B[1].reshape(3, 1), axis=1).T,
                         *np.append(d_W[0], d_B[0].reshape(3, 1), axis=1).T]).flatten()
    flat_grad = np.asarray([*grad[0].T, *grad[1].T, *grad[2].T]).flatten()
    for i in range(20):
        new_W = [(W[0] + epsilon * d_W[0]), (W[1] + epsilon * d_W[1]), (W[2] + epsilon * d_W[2])]
        func_with_epsilon, notIntresting = forward_pass(np.tanh, X, new_W, b, len(W), C)
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
    plt.close()


for i in range(20):
    test_grad_whole_network()

for i in range(l, -1, -1)