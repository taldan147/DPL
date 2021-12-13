import math

import torch
import LSTM_AE as AE
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os

parser = argparse.ArgumentParser(description="Arguments of MNIST AE")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=20, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=1, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=28, help="size of an input")
parser.add_argument('--dropout', type=float, default=0, help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=28, help="size of a seq")
parser.add_argument('--output_size', type=int, default=10, help="size of the output")

args =  parser.parse_args()

currDir = os.getcwd()
data = np.load(f"{currDir}/mnist_all.npz")


def parseData():
    trainData = []
    validateData = []
    trainLabel = []
    validateLabel = []
    labels = {"train0" : 0, "train1" : 1, "train2" : 2, "train3" : 3, "train4" : 4, "train5" : 5, "train6" : 6, "train7" : 7, "train8" : 8, "train9" : 9}
    keys = {k for k in data.keys() if k.startswith("train")}
    for key in keys:
        trainData.extend(data[key][:int(0.6 * len(data[key])), :])
        trainLabel.extend(np.repeat(labels[key],int(0.6 * len(data[key]))))
        validateData.extend(data[key][int(0.6 * len(data[key])):int(0.8 * len(data[key])), :])
        validateLabel.extend(np.repeat(labels[key],int(0.2 * len(data[key]))))
    trainData = np.asarray([np.split(x, 28) for x in trainData])/255
    validateData = np.asarray([np.split(x, 28) for x in validateData])/255
    return torch.from_numpy(trainData).type(torch.FloatTensor), torch.from_numpy(validateData).type(torch.FloatTensor), torch.from_numpy(np.asarray(trainLabel)), torch.from_numpy(np.asarray(validateLabel))

class MnistAE():
    def __init__(self):
        super(MnistAE, self).__init__()
        trainD, valData, trainL, valL = parseData()
        self.trainData = trainD
        self.validateData = valData
        self.trainLabel = trainL
        self.validateLabel = valL
        self.epochs = args.epochs
        self.batchs = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = AE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout, args.output_size)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), args.lr) if (
                    args.optimizer == "Adam") else torch.optim.SGD(self.AE.parameters(), lr=args.lr)

        print(f"using {self.device} as computing unit")


    def train(self):
        trainLoss = []
        validateLoss = []
        self.AE.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch}")
            currLoss = 0
            perm = np.random.permutation(len(self.trainData))
            for k in range(math.floor(len(self.trainData)/self.batchs)):
                print(f"this is iteration number {k+1} for epoch number {epoch+1}")
                indx = perm[k * self.batchs:(k + 1) * self.batchs]
                currX = self.trainData[indx]
                self.optimizer.zero_grad()
                currX.to(self.device)
                output = self.AE.forward(currX)
                loss = nn.MSELoss().forward(output, currX)
                loss.backward()
                self.optimizer.step()
                currLoss += loss.item()
            avgLoss = currLoss / len(self.trainData)
            trainLoss.append(avgLoss)
        return trainLoss

    def trainClassification(self):
        trainLoss = []
        validateLoss = []
        self.AE.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch}")
            currLoss = 0
            perm = np.random.permutation(len(self.trainData))
            for k in range(math.floor(len(self.trainData)/self.batchs)):
                print(f"this is iteration number {k+1} for epoch number {epoch+1}")
                indx = perm[k * self.batchs:(k + 1) * self.batchs]
                currX = self.trainData[indx]
                currLabel = self.trainLabel[indx]
                self.optimizer.zero_grad()
                currX.to(self.device)
                output = self.AE.forward(currX)
                loss = nn.CrossEntropyLoss().forward(output, currLabel)
                loss.backward()
                self.optimizer.step()
                currLoss += loss.item()
            avgLoss = currLoss / len(self.trainData)
            trainLoss.append(avgLoss)
        return trainLoss


    def reconstruct(self, data):
        data.to(self.device)
        return self.AE.forward(data)

    def plotNN(self):
        startLoss = time.perf_counter()
        loss = self.train()
        plt.figure()
        pixels = self.reconstruct(self.trainData).detach().squeeze().numpy()[0].reshape(28, 28)
        plt.title("Reconstructed")
        plt.imshow(pixels, cmap='gray')
        plt.show()

        pixels = self.trainData.detach().squeeze().numpy()[0].reshape(28, 28)
        plt.title("Original")
        plt.imshow(pixels, cmap='gray')
        plt.show()

        endLoss = time.perf_counter()
        endReconstruct = time.perf_counter()

        print("\nThe parameters of the NN are:")
        print(f"layers - {args.num_of_layers}\nepochs - {args.epochs}\nbatch size - {args.batch_size}\nlearning rate - {args.lr}\noptimizer - {args.optimizer}\n")
        print(f"the loss calc took {(endLoss-startLoss)/60} minutes")
        # print(f"the reconstruct calc took {(endReconstruct-endLoss)/60} minutes")
        print(f"overall it took {(endReconstruct-startLoss)/60} minutes")

    def plotClassification(self):
        startLoss = time.perf_counter()

        trainLoss = self.trainClassification()

        plt.figure()
        plt.title("Classification Loss")
        plt.plot(np.arange(self.epochs), trainLoss)

        endLoss = time.perf_counter()

        print(f"overall it took {(endLoss-startLoss)/60} minutes")


MnistAE().plotClassification()