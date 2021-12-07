import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
from Assignment1.Prod.task1_1 import soft_max, soft_max_regression, grad_soft_max, calculate_eta_vector
from Assignment1.Prod.task1_3 import classify, Cv_Peaks, Ct_Peaks, Ct_GMM , Cv_GMM, Ct_SwissRoll, Cv_SwissRoll, yv_Peaks, yt_Peaks, yt_SwissRoll, yt_GMM, yv_GMM, yv_SwissRoll
from Assignment1.Prod.neural_network import tanh_derivative

def forward_pass(f, X, W1, W2, b, l, C):
    keeper_X = [X]
    X_i = f(W1[0] @ X + b[0])  # for the first regular layer
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
    grad.append((tanh_derivative((W1[0] @ keeper_X[0] + B[0])) * deriv_by_x) @ keeper_X[0].T)  # for the first regular layer
    grad.append((tanh_derivative((W1[0] @ keeper_X[0] + B[0])) * deriv_by_x))  # for the first regular layer
    return grad

def test_jacobian_W_b():
    X = yt_SwissRoll[:,0].reshape(2,1)
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
    for i in range(20):
        func_with_epsilon = ((X + (W2 + epsilon*D_W2 ) @ np.tanh((W1+epsilon*D_W1) @ X + (b+epsilon*D_b))).T @ u).flatten()
        f_loss.append(abs(func_with_epsilon - func_result))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon *
                                               ((D_b.T @ db).flatten() + D_W1.flatten()@dw1.flatten()+
                                                D_W2.flatten()@dw2.flatten()))))
        epsilon *= 0.5
    plt.figure()
    plt.semilogy([i for i in range(20)], f_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad__loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Jacobian transpose test for the derivative of resnet layer by W and b')
    plt.legend()
    plt.show()

def test_jacobian_X():
    X = yt_SwissRoll[:,0].reshape(2,1)
    u = np.random.rand(2,1)
    b = np.random.rand(2).reshape(2, 1)
    W1 = np.random.rand(2, 2)
    W2 = np.random.rand(2, 2)
    D_X = np.random.rand(2,1)
    D_X = (1 / LA.norm(D_X)) * D_X
    f_loss = []
    grad__loss = []
    epsilon = 1
    func_result = ((X + W2 @ np.tanh(W1 @ X + b)).T @ u).flatten()
    dx = derive_by_X(X, W1, W2, b, u)
    for i in range(20):
        func_with_epsilon = (((X+epsilon*D_X) + W2 @ np.tanh(W1 @ (X+epsilon*D_X) + b)).T @ u).flatten()
        f_loss.append(abs(func_with_epsilon - func_result))
        grad__loss.append(abs(
            func_with_epsilon - func_result - (epsilon * (np.ndarray.flatten(D_X) @ np.ndarray.flatten(dx)))))
        epsilon *= 0.5
    plt.figure()
    plt.semilogy([i for i in range(20)], f_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad__loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.suptitle('Jacobian transpose test for the derivative of resnet layer by X')
    plt.legend()
    plt.show()

def d_mult_grad(grad, d_W1, d_W2, d_b):
    ans = grad[0].flatten() @ d_W1[2].flatten()
    for i in range(1,3):
        ans += grad[i][0].flatten() @ d_W1[2-i].flatten()
        ans += grad[i][1].flatten() @ d_W2[2-i].flatten()
        ans += np.sum(grad[i][2], axis=1) @ d_b[2-i]
    return ans

def test_grad_whole_network():
    X = yt_Peaks
    C = Ct_Peaks
    W1 = [np.random.rand(2,2),np.random.rand(2,2), np.random.rand(2,5)]
    W2 = [np.random.rand(2,2),np.random.rand(2,2)]
    b = [np.random.rand(2).reshape(2,1), np.random.rand(2).reshape(2,1)]
    d_W1 = [np.random.rand(2,2),np.random.rand(2,2), np.random.rand(2,5)]
    d_W2 = [np.random.rand(2,2),np.random.rand(2,2), np.random.rand(2,5)]
    d_B = [np.random.rand(2).reshape(2,1), np.random.rand(2).reshape(2,1)]
    soft_max_loss = []
    grad_soft_max_loss = []
    epsilon = 1
    func_result, keeper_X, no = forward_pass(np.tanh, X, W1, W2, b, len(W1), C)
    grad = back_propagation(keeper_X, W1, W2,b, len(W1), C)
    for i in range(20):
        new_W1 = [(W1[0]+epsilon*d_W1[0]), (W1[1]+epsilon*d_W1[1]), (W1[2]+epsilon*d_W1[2])]
        new_W2 = [(W2[0]+epsilon*d_W2[0]), (W2[1]+epsilon*d_W2[1])]
        new_B = [b[0] + epsilon * d_B[0], b[1] + epsilon * d_B[1]]
        func_with_epsilon, notIntresting, no2 = forward_pass(np.tanh, X, new_W1, new_W2, new_B, len(W1), C)
        soft_max_loss.append(abs(func_with_epsilon - func_result))
        grad_soft_max_loss.append(abs(func_with_epsilon - func_result - (epsilon * d_mult_grad(grad, d_W1, d_W2, d_B))))
        epsilon *= 0.5

    plt.figure()
    plt.semilogy([i for i in range(20)], soft_max_loss, label="Zero order approx")
    plt.semilogy([i for i in range(20)], grad_soft_max_loss, label="First order approx")
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('Gradient test for the whole Residual Network')
    plt.legend()
    plt.show()



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

def generate_list_of_matrix(num_of_matrix, shape):
    list = []
    for i in range(num_of_matrix):
        list.append(np.random.rand(shape[0],shape[1]))
    return list

def normalize_matrix(list_w):
    for i in range(len(list_w)):
        list_w[i] = list_w[i] / LA.norm(list_w[i])
    return list_w

def SGD_ResNet(back_prop, X, W1, W2, b, c, epoch, batch, X_valid, C_valid):
    lr = 0.5
    success_percentages = [calculate_success_Res_Net(X, W1, W2, b, c)]
    success_percentages_valid = [calculate_success_Res_Net(X_valid, W1, W2, b, C_valid)]
    for i in range(epoch):
        perm = np.random.permutation(len(X[0]))
        # lr = 1/(math.sqrt(1+i))
        if i == 100:
            lr *=0.1
        for k in range(math.floor(len(X[0])/batch)):
            indx = perm[k*batch:(k+1)*batch]
            currX = X[:, indx]
            currC = c[:, indx]
            no1, keeperX, no2 = forward_pass(np.tanh, currX, W1,W2, b, len(W1), currC)
            gradK = back_prop(keeperX, W1,W2, b, len(W1), currC)
            W1,W2,b = update_params(W1,W2,b, gradK, lr)
        success_percentages.append(calculate_success_Res_Net(X, W1,W2,b, c))
        success_percentages_valid.append(calculate_success_Res_Net(X_valid, W1,W2,b, C_valid))
    return success_percentages, success_percentages_valid

def test_ResNet(X_train, X_valid, C_train, C_valid, W1, W2, b, epoch, batch, data):
    success_percentages_train, success_percentages_validation = SGD_ResNet(back_propagation, X_train, W1, W2, b, C_train, epoch, batch, X_valid, C_valid)
    plt.plot(np.arange(len(success_percentages_train)), [x * 100 for x in success_percentages_train],
             label='train')
    plt.plot(np.arange(len(success_percentages_validation)), [x * 100 for x in success_percentages_validation],
             label='validation')
    # plt.title('Success percentages of NN for %s,\n' % data + r' lr: $\frac{1}{\sqrt{epoch}}$, batch: %s' % (batch))
    plt.title('Success percentages of ResNet for ' + data + ',\n lr: 0.5, batch: ' + str(batch))
    print("avg train success percentage of ", data, ":",
          success_percentages_train[99])
    print("avg validation success percentage of ", data, ":",
          success_percentages_validation[99])
    print("avg train success percentage of ", data, ":",
          np.mean(np.delete(np.asarray(success_percentages_train), np.arange(5))))
    print("avg validation success percentage of ", data, ":",
          np.mean(np.delete(np.asarray(success_percentages_validation), np.arange(5))))
    plt.xlabel('epoch')
    plt.ylabel('success percentage')
    plt.legend()
    plt.show()

def test_ResNet_swissroll():
    W1 = [np.random.rand(10, 2), np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(11, 2)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10, 10), np.random.rand(10, 10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 1)]
    epoch = 200
    batch = 100
    test_ResNet(yt_SwissRoll, yv_SwissRoll, Ct_SwissRoll, Cv_SwissRoll, W1, W2, b, epoch, batch, "SwissRoll")



def test_ResNet_peaks():
    W1 = [np.random.rand(10,2), np.random.rand(10,10),np.random.rand(10,10), np.random.rand(11,5)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10,10),np.random.rand(10,10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)]
    epoch = 200
    batch = 100
    test_ResNet(yt_Peaks, yv_Peaks, Ct_Peaks, Cv_Peaks, W1, W2, b, epoch, batch, "Peaks")

def test_ResNet_GMM():
    W1 = [np.random.rand(10,5), np.random.rand(10,10),np.random.rand(10,10), np.random.rand(11,5)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10,10),np.random.rand(10,10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)]
    epoch = 300
    batch = 100
    test_ResNet(yt_GMM, yv_GMM, Ct_GMM, Cv_GMM, W1, W2, b, epoch, batch, "GMM")

def test_ResNet_GMM_200():
    W1 = [np.random.rand(10,5), np.random.rand(10,10),np.random.rand(10,10), np.random.rand(11,5)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10,10),np.random.rand(10,10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)]
    epoch = 300
    batch = 100
    X, C = pick_sample(yt_GMM, Ct_GMM, 200)
    test_ResNet(X, yv_GMM, C, Cv_GMM, W1, W2, b, epoch, batch, "GMM")

def test_ResNet_peaks_200():
    W1 = [np.random.rand(10,2), np.random.rand(10,10),np.random.rand(10,10), np.random.rand(11,5)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10,10),np.random.rand(10,10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10,1),np.random.rand(10,1),np.random.rand(10,1)]
    epoch = 200
    batch = 100
    X, C = pick_sample(yt_Peaks, Ct_Peaks, 200)
    test_ResNet(X, yv_Peaks, C, Cv_Peaks, W1, W2, b, epoch, batch, "Peaks")

def test_ResNet_swissroll_200():
    W1 = [np.random.rand(10, 2), np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(11, 2)]
    W1 = normalize_matrix(W1)
    W2 = [np.random.rand(10, 10), np.random.rand(10, 10)]
    W2 = normalize_matrix(W2)
    b = [np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 1)]
    epoch = 200
    batch = 100
    X, C = pick_sample(yt_SwissRoll, Ct_SwissRoll, 200)
    test_ResNet(X, yv_SwissRoll, C, Cv_SwissRoll, W1, W2, b, epoch, batch, "SwissRoll")

def pick_sample(X, c, m):
    perm = np.random.permutation(len(X[0]))
    indx = perm[0: m]
    sampleX = X[:, indx]
    samplec = c[:, indx]
    return sampleX, samplec

# test_ResNet_swissroll()
# test_ResNet_GMM()
# test_ResNet_peaks()
# test_grad_whole_network()
# test_ResNet_peaks_200()
# test_ResNet_GMM_200()
test_ResNet_swissroll()

# test_jacobian_W_b()
# test_jacobian_X()