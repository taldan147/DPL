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

def soft_max_regression(X, W, C):
    sum=0
    for i in range(len(X)):
        tmp = calc_divider(X, W)
        sum += C[i].transpose() @ (np.log(np.exp(np.ndarray.transpose(X) @ W[i]) / tmp))
    return sum / len(X)

def calc_divider(X, W):
    sum=np.zeros(len(X[0]))
    for i in range(len(W)):
        sum += np.exp(np.ndarray.transpose(X) @ W[i])
    return sum

def soft_max_gradient(X, W, C, p):
    return (1/len(X)) * (X @ (np.exp(X.transpose() @ W[p]) / calc_divider(X, W) - C[p]))


def SGD(objective,grad_obj, max_iter, X, y, w):
    grads_norms = [LA.norm(y)]
    lr = 1
    batches = 2
    for epoch in range(max_iter):
        indx = np.random.permutation(len(X))
        lr = lr / (math.sqrt(epoch+1))
        # if epoch % 100 == 0:
        #     lr = lr * 0.1
        for i in range(math.floor(len(X)/batches)):
            currIndx = indx[i*batches:(i+1)*batches]
            currX = np.asarray([X[j] for j in currIndx])
            # g_k = (1/len(currX)) * np.ndarray.transpose(currX) @ ((currX @ w) - np.asarray([y[j] for j in currIndx])) + (0.002 * w)
            # g_k = (sum([grad_obj(X, y, w, j) for j in currIndx]) / 1/len(currX))
            g_k = grad_obj(currX, np.asarray([y[j] for j in currIndx]), w) / len(currX)
            w = w - (lr*g_k)
        # grads_norms.append(LA.norm((1/len(X)) * np.ndarray.transpose(X) @ ((X @ w) - y) + (0.002 * w)))
        grads_norms.append(LA.norm((1/len(X)) * grad_obj(X, y, w)))
    return w, grads_norms

def test_SGD_with_LS_example():
    X = np.random.rand(300, 200)
    U, S, V = LA.svd(X, full_matrices=True)
    to_fill = np.exp(0.5* np.random.rand(200))
    S = np.zeros((300, 200))
    np.fill_diagonal(S, to_fill)
    X = U @ np.asarray(S) @ np.ndarray.transpose(V)
    w = np.zeros(200)
    y = (X @ np.random.rand(200))+(0.05 * np.random.rand(300))
    objective = lambda X, y, w : (1/(2*len(X))) * LA.norm((X @ w) - y)
    grad_obj_i = lambda X, y, w, i : (X[i] @ w - y[i]) * X[i]
    w, grads_norms = SGD(objective, grad_obj_i, 200, X, y, w)
    plot_graphs(grads_norms)


def LS_grad(X, y, w):
    sum = np.zeros(len(w))
    for i in range(len(X)):
        sum += ((X[i].transpose() @ w) - y[i])* X[i]
    return sum
def test_SGD_LS_2():
    X = np.asarray([[-1,-1,1], [1,3,3],[-1,-1,5],[1,3,7]])
    y = np.asarray([0,23,15,39])
    w = np.zeros(3)
    # w = np.asarray([1,3,4])
    obj = lambda X, y, w : LA.norm((X@w) - y)**2
    w, grads_norms = SGD(obj, LS_grad, 200, X, y, w)
    plot_graphs(grads_norms)

def plot_graphs(grads_norms):
    plt.semilogy(grads_norms, label='grads_norms' )
    plt.ylim(bottom=10**-4)
    plt.suptitle("SGD")
    plt.xlabel("iteration")
    plt.ylabel('f_values')
    plt.legend()
    plt.show()

# test_SGD_with_LS_example()
# test_SGD_LS_2()

# a = np.zeros((5,10),int)
# x = [1,2,3,4,5]
# np.fill_diagonal(a, x)
# print(a)

def calc_grad_matrix(X, W, C):
    grad_matrix = []
    for i in range(len(W)):
        grad_matrix.append(soft_max_gradient(X,W,C,i))
    return np.asarray(grad_matrix)


def test_grad(X,C,W , epsilon, d, maxIter):

    loss_with_grad = []
    loss = []
    epsilons = []
    # DF = []
    # DF_grad = []
    for i in range(maxIter):
        epsilons.append(epsilon)
        f_eps= soft_max_regression(X, W + epsilon*d , C)
        f = soft_max_regression(X, W , C)
        grad_with = calc_grad_matrix(X, W, C)
        loss.append(abs(f_eps-f))
        loss_with_grad.append(abs(f_eps-f-(epsilon*(1/len(W)) * np.ndarray.flatten(d) @ np.ndarray.flatten(grad_with))))
        epsilon *=0.5
        # if len(loss) >1:
        #     DF.append(loss[i-1] / loss[i])
        #     DF_grad.append((loss_with_grad[i-1]/loss_with_grad[i]))

    return loss, loss_with_grad, epsilons


def plot_grad_test():
    X = yt
    c = Ct
    w = np.random.rand(20000,2)
    d = np.random.rand(20000,2)
    loss, loss_with_grad, epsilons = test_grad(X, c, w, 0.5, d, 20)
    plt.figure();
    plt.semilogy(epsilons, loss, label="gradient test : loss");
    plt.semilogy(epsilons, loss_with_grad, label="loss_grad");
    plt.xlabel('epsilons')
    plt.ylabel('Decrease factor')
    plt.legend();
    plt.show()

plot_grad_test()


