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

def forward_pass(f, X, W1, W2, b, l, C):
    keeper_X = [X]
    X_i = f(W1[0] @ X + b[0])
    keeper_X.append(X_i)
    for i in range(l - 2):
        X_i = X_i + W2[i] @ f(W1[i+1] @ X_i + b[i])
        keeper_X.append(X_i)
    keeper_X[l-1] = np.vstack([keeper_X[l-1], np.ones(len(keeper_X[l-1][0]))])
    return soft_max_regression(keeper_X[l-1], C, W1[l-1]), keeper_X, soft_max(keeper_X[l-1], W1[l-1])

def derive_by_X(X, W1, W2,b,v):
    return v + W1.T @ derive_by_b(X,W1,W2,b,v)

def derive_by_b(X, W1, W2,b ,v):
    return tanh_derivative((W1 @ X) + b) * (W2.T @ v)

def derive_by_W1(X, W1, W2,b,v):
    return derive_by_b(X,W1,W2,b,v) @ X.T

def derive_by_W2(X,W1,W2,b,v):
    return v @ np.tanh(W1 @ X + b).T

def grad_soft_max_by_X(X,W,C):
    eta = calculate_eta_vector(X,W)
    return (1/len(X[0])) * (W @ ((np.exp(W.T@X - eta) / np.sum(np.exp(W.T @ X - eta), axis=0)) - C))

def back_propagation(keeper_X, W1, W2,B, l, C):
    grad = [grad_soft_max(keeper_X[l-1], W1[l-1], C)]
    deriv_by_x = grad_soft_max_by_X(keeper_X[l-1],W1[l-1],C)
    deriv_by_x = np.delete(deriv_by_x, len(deriv_by_x)-1, 0)
    for i in range(l-2, 0, -1):
        dw1 = derive_by_W1(keeper_X[i], W1[i], W2[i-1], B[i-1], deriv_by_x)
        dw2 = derive_by_W2(keeper_X[i], W1[i], W2[i-1], B[i-1], deriv_by_x)
        db = derive_by_b(keeper_X[i], W1[i], W2[i-1], B[i-1], deriv_by_x)
        curr_grad = [dw1, dw2, db]
        grad.append(curr_grad)
        deriv_by_x = derive_by_X(keeper_X[i], W1[i], W2[i-1], B[i-1], deriv_by_x)
    grad.append((tanh_derivative((W1[0] @ keeper_X[0] + B[0])) * deriv_by_x) @ keeper_X[0].T)
    grad.append((tanh_derivative((W1[0] @ keeper_X[0] + B[0])) * deriv_by_x))
    return grad

def tanh_derivative(X):
    return np.ones(np.shape(X)) - np.power(np.tanh(X), 2)

def test_jacobian():
    # X = yt
    # X = np.vstack([X, np.ones(len(X[0]))])
    # X = np.append(yt[:,0],1).reshape(3,1)
    X = yt[:,0].reshape(2,1)
    u = np.random.rand(2).reshape(2,1)
    b = np.random.rand(2).reshape(2,1)
    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,2)
    D_W1 = np.random.rand(2,2)
    D_W2 = np.random.rand(2,2)
    D_b = np.random.rand(2).reshape(2,1)
    D_W1 = (1 / LA.norm(D_W1)) * D_W1
    D_W2 = (1 / LA.norm(D_W2)) * D_W2
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = ((X + W2 @ np.tanh(W1 @ X + b)).T @ u).flatten()
    dw1 = derive_by_W1(X, W1, W2,b, u)
    db = derive_by_b(X, W1, W2,b, u)
    dw2 = derive_by_W2(X, W1, W2,b, u)
    # grad = np.append(dw, np.reshape(db, (2,1)), axis=1)
    # D = np.append(D_W, np.reshape(D_b, (2,1)), axis=1)
    for i in range(20):
        # func_with_epsilon = np.tanh(W @ X + (b+D_b*epsilon)) @ u
        func_with_epsilon = ((X + (W2 + epsilon*D_W2 ) @ np.tanh((W1+epsilon*D_W1) @ X + (b+epsilon*D_b))).T @ u).flatten()
        f_loss.append(abs(func_with_epsilon - func_result))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon *
                                               ((D_b.T @ db).flatten()+ D_W1.flatten()@dw1.flatten()+
                                                D_W2.flatten()@dw2.flatten()))))
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

def d_mult_grad(grad, d_W1, d_W2, d_b):
    ans = d_W1[3].flatten() @ grad[0].flatten()
    for i in range(1,3):
        ans += d_W1[3-i].flatten() @ grad[i][0].flatten()
        ans += d_W2[2-i].flatten() @ grad[i][1].flatten()
        ans += (d_b[3-i].T @ np.sum(grad[i][2], axis=1).reshape(len(d_b[3-i]),1)).flatten()
    ans += d_W1[0].flatten() @ grad[3].flatten()
    ans += (d_b[0].T @ np.sum(grad[4], axis=1).reshape(len(d_b[0]),1)).flatten()

    return ans

def test_grad_whole_network():
    X = yt
    # X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    # W1 = generate_list_of_matrix(2,[2,2])
    # W1.append(np.random.rand(2,5))
    # W2 = generate_list_of_matrix(2,[2,2])
    # b = generate_list_of_matrix(2,[2,1])
    W1 = [np.random.rand(10, 2), np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(11, 5)]
    # W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10, 10), np.random.rand(10, 10)]
    # W2 = normalize_matrix(W2)
    b = [np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 1)]
    # W1 = [np.random.rand(2,2),np.random.rand(2,2), np.random.rand(2,5)]
    # W2 = [np.random.rand(2,2),np.random.rand(2,2)]
    # b = [np.random.rand(2).reshape(2,1), np.random.rand(2).reshape(2,1)]
    d_W1 = [np.random.rand(10, 2), np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(11, 5)]
    d_W2 = [np.random.rand(10, 10), np.random.rand(10, 10)]
    d_B = [np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 1)]
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result, keeper_X, no = forward_pass(np.tanh, X, W1, W2, b, len(W1), C)
    grad = back_propagation(keeper_X, W1, W2,b, len(W1), C)
    for i in range(20):
        new_W1 = [(W1[0]+epsilon*d_W1[0]), (W1[1]+epsilon*d_W1[1]), (W1[2]+epsilon*d_W1[2]), (W1[3]+epsilon*d_W1[3])]
        new_W2 = [(W2[0]+epsilon*d_W2[0]), (W2[1]+epsilon*d_W2[1])]
        new_B = [b[0] + epsilon * d_B[0], b[1] + epsilon * d_B[1], b[2] + epsilon * d_B[2]]
        func_with_epsilon, notIntresting, no2 = forward_pass(np.tanh, X, new_W1, new_W2, new_B, len(W1), C)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon - func_result - (epsilon * d_mult_grad(grad, d_W1, d_W2, d_B))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Grad test in semilogarithmic plot')
    plt.legend()
    plt.show()

test_grad_whole_network()



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

def SGD_ResNet(back_prop, X, W1, W2, b, c, epoch, batch):
    # norms = []
    # lr = 0.1
    # batch = 6000
    # success_percentages = [soft_max_regression(X,c,w)]
    success_percentages = [calculate_success_Res_Net(X, W1, W2, b, c)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        lr = 1/(math.sqrt(1+i))
        # if i % 50 == 0:
        #     lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currC = c[:, indx]
            no1, keeperX, no2 = forward_pass(np.tanh, currX, W1,W2, b, len(W1), currC)
            gradK = back_prop(keeperX, W1,W2, b, len(W1), currC)
            W1,W2,b = update_params(W1,W2,b, gradK, lr)
        # norms.append(LA.norm((1/len(X)) * X.transpose() @ ((X@w) -c) + 0.01*w))
        # success_percentages.append(soft_max_regression(X,c,w))
        success_percentages.append(calculate_success_Res_Net(X, W1,W2,b, c))
    return success_percentages

def update_params(W1,W2,b, grad, lr):
    len_grad = len(grad)
    l = len(W1)
    W1[l-1] -= lr*grad[0]
    W1[0] -= grad[len_grad-2]
    b[0] -= np.sum(grad[len_grad-1], axis=1).reshape(len(grad[len_grad-1]),1)
    for i in range(1, l-1):
        W1[l-i-1] -= lr*grad[i][0]
        W2[l-i-2] -= lr*grad[i][1]
        b[l-i-1] -= lr*np.sum(grad[i][2], axis=1).reshape(len(b[l-i-1]), 1)
    return W1,W2,b

def calculate_success_Res_Net(X, W1, W2,b, C):
    no1, no2, SMresult = forward_pass(np.tanh, X, W1, W2, b, len(W1), C)
    classified = classify(X, W1[len(W1)-1], SMresult)
    return 1 - np.sum(abs(C - classified)) / (2 * len(X[0]))

def test_NN_swissroll():
    X = yt
    X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    # W1 = generate_list_of_matrix(2,[2,2])
    # W1.append(np.random.rand(2,5))
    # W2 = generate_list_of_matrix(2,[2,2])
    # b = generate_list_of_matrix(2,[2,1])
    W1 = [np.random.rand(10, 3), np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(10, 2)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10, 10), np.random.rand(10, 10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10, 1), np.random.rand(10, 1)]
    W1_valid = W1.copy()
    W2_valid = W2.copy()
    X_valid = yv
    X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    C_valid = Cv
    epoch = 100
    success_percentages_train = SGD_ResNet(back_propagation, X, W1, W2, b, C, epoch, 100)
    success_percentages_validation = SGD_ResNet(back_propagation, X_valid, W1_valid, W2_valid, b, C_valid, epoch, 100)
    plt.plot(np.arange(len(success_percentages_train)), [x * 100 for x in success_percentages_train],
             label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), [x * 100 for x in success_percentages_validation],
             label='success percentage for validation per epoch')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

def generate_list_of_matrix(num_of_matrix, shape):
    list = []
    for i in range(num_of_matrix):
        list.append(np.random.rand(shape[0],shape[1]))
    return list

def normalize_matrix(list_w):
    for i in range(len(list_w)):
        list_w[i] = list_w[i] / LA.norm(list_w[i])
    return list_w

def test_NN_peaks():
    X = yt
    # X = np.vstack([X, np.ones(len(X[0]))])
    C = Ct
    # W1 = generate_list_of_matrix(2,[2,2])
    # W1.append(np.random.rand(2,5))
    # W2 = generate_list_of_matrix(2,[2,2])
    # b = generate_list_of_matrix(2,[2,1])
    W1 = [np.random.rand(10,2), np.random.rand(10,10),np.random.rand(10,10), np.random.rand(11,5)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10,10),np.random.rand(10,10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)]
    W1_valid = W1.copy()
    W2_valid = W2.copy()
    X_valid = yv
    # X_valid = np.vstack([X_valid, np.ones(len(X_valid[0]))])
    C_valid = Cv
    epoch = 100
    success_percentages_train = SGD_ResNet(back_propagation, X, W1,W2, b, C, epoch, 100)
    success_percentages_validation = SGD_ResNet(back_propagation, X_valid, W1_valid,W2_valid, b, C_valid, epoch, 100)
    plt.plot(np.arange(len(success_percentages_train)), [x*100 for x in success_percentages_train], label='success percentage for train per epoch')
    plt.plot(np.arange(len(success_percentages_validation)), [x*100 for x in success_percentages_validation], label='success percentage for validation per epoch')
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

# test_NN_peaks()
# test_NN_swissroll()