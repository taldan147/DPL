import math

import torch
import LSTM_AE as AE
import SyntethicDataGen as Gen
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description="Arguments of Toy AE")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=100, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=1, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=1, help="size of an input")
parser.add_argument('--dropout', type=float, default=0, help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=50, help="size of a seq")
parser.add_argument('--output_size', type=int, default=1, help="size of the output")
parser.add_argument('--grad_clip', type=int, default=None, help="gradient clipping value")
args =  parser.parse_args()

class ToyAE():
    def __init__(self, train, validation, test, lr, grad_clip, hs_size):
        super(ToyAE, self).__init__()
        self.trainData = train
        self.validateData = validation
        self.testData = test
        self.epochs = args.epochs
        self.batchs = args.batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = AE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, hs_size, args.dropout, args.output_size)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), lr) if (args.optimizer == "Adam")  else torch.optim.SGD(self.AE.parameters(), lr)

        print(f"using {self.device} as computing unit")

    def train(self):
        trainLoss = []
        validateLoss = []
        model = self.AE.to(self.device)
        mse = nn.MSELoss().to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch}")
            currLoss = 0
            perm = np.random.permutation(len(self.trainData))
            for k in range(math.floor(len(self.trainData)/self.batchs)):
                print(f"this is iteration number {k+1}/{math.floor(len(self.trainData)/self.batchs)} for epoch number {epoch+1}/{self.epochs}")
                indx = perm[k * self.batchs:(k + 1) * self.batchs]
                currX = self.trainData[indx].to(self.device)
                self.optimizer.zero_grad()
                output = model(currX)
                loss = mse.forward(output, currX)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                currLoss += loss.item()
            avgLoss = currLoss / (math.floor(len(self.trainData)/self.batchs))
            validateData = self.validateData.to(self.device)
            currLoss = mse.forward(model(validateData), validateData)
            validateLoss.append(currLoss.item())
            trainLoss.append(avgLoss)
        return trainLoss, validateLoss


    def reconstruct(self, data):
        return self.AE.to(self.device).forward(data.to(self.device))

    def plotNN(self, savePlt=False):

        startLoss = time.perf_counter()
        trainLoss, validLoss = self.train()
        plt.figure()
        plt.title("Loss of train and validation - Toys")
        plt.plot(np.arange(self.epochs), trainLoss, label='train')
        plt.plot(np.arange(self.epochs), validLoss, label='validation')
        plt.legend()
        if savePlt:
            plt.savefig(f"Plots/TrainValidateLoss.png")
        plt.show()

        endLoss = time.perf_counter()

        reconstruct = self.reconstruct(self.trainData).detach().cpu().squeeze().numpy()

        for i in range(5):
            plt.figure()
            plt.title("Original signal vs Reconstructed signal")
            plt.plot(reconstruct[i], label="reconstructed")
            plt.plot(self.trainData[i], label="original")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            if savePlt:
                plt.savefig(f"Plots/ToyReconstruction.png")
            plt.show()

        endReconstruct = time.perf_counter()

        print("The parameters of the NN are:")
        print(f"layers - {args.num_of_layers}\nepochs - {self.epochs}\nbatch size - {args.batch_size}\nlearning rate - {self.lr}\noptimizer - {self.optimizer}\n")
        print(f"the loss calc took {(endLoss-startLoss)/60} minutes")
        print(f"the reconstruct calc took {(endReconstruct-endLoss)/60} minutes")
        print(f"overall it took {(endReconstruct-startLoss)/60} minutes")

data = Gen.genData()
trainData = data[:6000]
validateData = data[6000:8000]
testData = data[8000:]

def grid_search():
    lr_arr = [0.01, 0.001, 0.0001]
    hs_size_arr = [8, 16, 32]
    grad_clip_arr = [None, 1, 10]

    params_loss_keeper = {}
    best_params = {'lr': None, 'hs_size': None, 'grad_clip': None}
    best_loss = np.Inf
    for lr in lr_arr:
        for hs_size in hs_size_arr:
            for grad_clip in grad_clip_arr:
                train_loss, val_loss = ToyAE(trainData, validateData, testData, lr, grad_clip, hs_size).train()
                curr_loss = val_loss[-1]
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_params = {'lr': lr, 'hs_size': hs_size, 'grad_clip': grad_clip}
                params_loss_keeper.update({f'lr: {lr}, hs_size: {hs_size}, grad_clip: {grad_clip}:': curr_loss})
    print(f'Best parameters found: {best_params}')
    print(f'Best Validation Loss: {best_loss}')
    print(f'Parameters loss: {params_loss_keeper}')
    ToyAE(trainData, validateData, testData, best_params['lr'], best_params['grad_clip'], best_params['hs_size']).plotNN(savePlt=False)

grid_search()
lr = 0.01
hs_size = 40
grad_clip = 1
# ToyAE(trainData, validateData, testData, lr, grad_clip, hs_size).plotNN(savePlt=False)