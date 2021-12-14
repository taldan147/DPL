import math

import torch
import LSTM_AE as AE
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torchvision
import os

parser = argparse.ArgumentParser(description="Arguments of MNIST AE")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=128, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=1, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=28, help="size of an input")
parser.add_argument('--dropout', type=float, default=0, help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=28, help="size of a seq")
parser.add_argument('--output_size', type=int, default=28, help="size of the output")

args = parser.parse_args()

currDir = os.getcwd()
netDir = f"{currDir}/SavedNets/MNIST"
classifyDir = f"{currDir}/SavedNets/MNIST"


def parseData():
    trainData = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

    testData = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=True)
    return trainLoader, testLoader, trainData

class MnistAE():
    def __init__(self):
        super(MnistAE, self).__init__()
        trainLoader, testLoader, trainSet = parseData()
        self.trainData = trainLoader
        self.testData = testLoader
        self.trainSet = trainSet
        self.epochs = args.epochs
        self.batchs = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = AE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout, args.output_size)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), args.lr) if (
                    args.optimizer == "Adam") else torch.optim.SGD(self.AE.parameters(), lr=args.lr)

        print(f"using {self.device} as computing unit")


    def train(self):
        trainLoss = []
        self.AE.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch+1}")
            currLoss = 0
            for ind, (img, label) in enumerate(self.trainData):
                print(f"this is iteration number {ind+1} for epoch number {epoch+1}")
                currX = img.squeeze()
                self.optimizer.zero_grad()
                currX.to(self.device)
                output = self.AE.forward(currX)
                loss = nn.MSELoss().forward(output, currX)
                loss.backward()
                self.optimizer.step()
                currLoss += loss.item()
            avgLoss = currLoss / len(self.trainData)
            trainLoss.append(avgLoss)
        # torch.save(self.AE.state_dict(), netDir)
        print("Finished training. Saving Net")
        return trainLoss

    def trainClassification(self):
        trainLoss = []
        self.AE.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch+1}")
            currLoss = 0
            for ind, (img, label) in enumerate(self.trainData):
                print(f"this is iteration number {ind+1} for epoch number {epoch+1}")
                currX = img.squeeze()
                self.optimizer.zero_grad()
                currX.to(self.device)
                output = self.AE.forward(currX)
                loss = nn.CrossEntropyLoss().forward(input=output, target=label)
                loss.backward()
                self.optimizer.step()
                currLoss += loss.item()
            avgLoss = currLoss / len(self.trainData)
            trainLoss.append(avgLoss)
        # torch.save(self.AE.state_dict(), netDir)
        print("Finished training. Saving Net")
        return trainLoss


    def reconstruct(self, data):
        return self.AE.forward(data.type(torch.FloatTensor))

    def plotNN(self):
        startLoss = time.perf_counter()
        loss = self.train()
        plt.figure()
        pixels = self.reconstruct(self.trainSet.data).detach().squeeze().numpy()[0].reshape(28, 28)
        plt.title("Reconstructed")
        plt.imshow(pixels, cmap='gray')
        plt.show()


        plt.title("Loss")
        plt.plot(np.arange(self.epochs), loss)
        plt.show()

        pixels = self.trainSet.data.detach().squeeze().numpy()[0].reshape(28, 28)
        plt.title("Original")
        plt.imshow(pixels, cmap='gray')
        plt.show()

        endLoss = time.perf_counter()
        endReconstruct = time.perf_counter()

        print("\nThe parameters of the NN are:")
        print(f"layers - {args.num_of_layers}\nepochs - {args.epochs}\nbatch size - {args.batch_size}\nlearning rate - {args.lr}\noptimizer - {args.optimizer}\n")
        print(f"the loss calc took {(endLoss-startLoss)/60} minutes")
        print(f"the reconstruct calc took {(endReconstruct-endLoss)/60} minutes")
        print(f"overall it took {(endReconstruct-startLoss)/60} minutes")

    def plotClassification(self):
        startLoss = time.perf_counter()

        trainLoss = self.trainClassification()

        plt.figure()
        plt.title("Classification Loss")
        plt.plot(np.arange(self.epochs), trainLoss)

        endLoss = time.perf_counter()

        print(f"overall it took {(endLoss-startLoss)/60} minutes")


MnistAE().plotNN()