import math

import torch
import LSTM_AE as AE
import MnistClassifier as AEC
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torchvision
import os

parser = argparse.ArgumentParser(description="Arguments of MNIST AE")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--epochs', type=int, default=2, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=50, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=3, help="num of layers")
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
    trainData = torchvision.datasets.MNIST(root=f'{currDir}/data', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

    testData = torchvision.datasets.MNIST(root=f'{currDir}/data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=True)
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
        self.AEC = AEC.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout, args.output_size)
        self.optimizer = torch.optim.Adam(self.AEC.parameters(), args.lr) if (
                    args.optimizer == "Adam") else torch.optim.SGD(self.AEC.parameters(), lr=args.lr)

        print(f"using {self.device} as computing unit")


    def train(self):
        trainLoss = []
        self.AE.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch+1}")
            currLoss = 0
            for ind, (img, label) in enumerate(self.trainData):
                print(f"this is iteration number {ind+1}/{len(self.trainData)} for epoch number {epoch+1}/{args.epochs}")
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
        accuracy = []
        print("Starting Train")
        self.AEC.to(self.device)
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch+1}")
            currLoss = 0
            currAcc = 0
            for ind, (img, label) in enumerate(self.trainData):
                print(f"this is iteration number {ind+1}/{len(self.trainData)} for epoch number {epoch+1}/{args.epochs}")
                currX = img.squeeze()
                self.optimizer.zero_grad()
                currX.to(self.device)
                output, classed = self.AEC(currX)
                lossClass = nn.CrossEntropyLoss().forward(input=classed.squeeze(), target=label)
                loss = nn.MSELoss().forward(output, currX)
                totalLoss = loss + lossClass
                totalLoss.backward()
                self.optimizer.step()
                currLoss += totalLoss.item()
                currAcc += self.accuracy(classed, label)
            avgLoss = currLoss / len(self.trainData)
            avgACC = currAcc / len(self.trainData)
            accuracy.append(avgACC)
            trainLoss.append(avgLoss)
        # torch.save(self.AE.state_dict(), netDir)
        print("Finished training. Saving Net")
        return trainLoss, accuracy

    def reconstruct(self, data):
        return self.AE(data.type(torch.FloatTensor))

    def reconstructClass(self, data):
        return self.AEC(data.type(torch.FloatTensor))

    def plotNN(self):

        startLoss = time.perf_counter()
        loss = self.train()
        dataIter = iter(self.testData)
        figure, labels = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.imshow(figure, cmap='gray')
        plt.show()

        reconed = self.reconstructClass(torch.unsqueeze(figure, 0))
        plt.title("Reconstructed")
        plt.imshow(reconed.detach().squeeze().numpy(), cmap='gray')
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
        plt.show()

        dataIter = iter(self.testData)
        figure, labels = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.imshow(figure, cmap='gray')
        plt.show()

        reconed, label = self.reconstructClass(torch.unsqueeze(figure, 0))
        fixedLabel = np.argmax(label.squeeze().detach().numpy(), axis=1)
        print(fixedLabel)
        plt.title("Reconstructed")
        plt.imshow(reconed.detach().squeeze().numpy(), cmap='gray')
        plt.show()

        endLoss = time.perf_counter()
        print(f"overall it took {(endLoss-startLoss)/60} minutes")


    def accuracy(self, predict, labels):
        newPredict = np.argmax(predict.squeeze().detach().numpy(), axis=1)
        return np.mean(newPredict != labels.detach().numpy())


MnistAE().plotClassification()