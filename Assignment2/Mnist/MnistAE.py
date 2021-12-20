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
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="Arguments of MNIST AE")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--epochs', type=int, default=3, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=400, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=3, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=28, help="size of an input")
parser.add_argument('--dropout', type=float, default=0,  help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=28, help="size of a seq")
parser.add_argument('--output_size', type=int, default=28, help="size of the output")
parser.add_argument('--pixel_output_size', type=int, default=1, help="size of the output PbP")
parser.add_argument('--pixel_seq_size', type=int, default=784, help="size of the seq PbP")
parser.add_argument('--pixel_input_size', type=int, default=1, help="size of the input PbP")

args = parser.parse_args()

currDir = f"{os.getcwd()}/Mnist"
netDir = f"{currDir}/SavedNets/Net.pt"
classifyDir = f"{currDir}/SavedNets/Classify.pt"
pixelDir = f"{currDir}/SavedNets/Pixel.pt"


def parseData():

    transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
    trainData = torchvision.datasets.MNIST(root=f'{currDir}/data', train=True, download=True,
                                           transform=transform)

    testData = torchvision.datasets.MNIST(root=f'{currDir}/data', train=False, download=True,
                                          transform=transform)

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
        self.AE = AE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout,
                            args.output_size)
        self.AEC = AEC.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout,
                              args.output_size)
        self.pixel_AEC = AEC.LSTMAE(args.pixel_input_size, args.num_of_layers, args.pixel_seq_size, args.hidden_size,
                                    args.dropout, args.pixel_output_size)
        self.optimizer = torch.optim.Adam(self.pixel_AEC.parameters(), args.lr) if (
                args.optimizer == "Adam") else torch.optim.SGD(self.AEC.parameters(), lr=args.lr)  #TODO define different oprimizer for each LSTM

        print(f"using {self.device} as computing unit")

    def train(self, saveNet):
        trainLoss = []
        self.AE.to(self.device)
        print("Starting Train")
        if saveNet:
            print("Will save net!")

        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch + 1}")
            currLoss = 0
            for ind, (img, label) in enumerate(self.trainData):
                print(
                    f"this is iteration number {ind + 1}/{len(self.trainData)} for epoch number {epoch + 1}/{args.epochs}")
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

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

        return trainLoss

    def trainClassification(self, useRows, saveNet):
        trainLoss = []
        accuracy = []
        print("Starting Train")
        if saveNet:
            print("Will save net!")
        NN = self.AEC if (useRows) else self.pixel_AEC
        NN.to(self.device)

        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch + 1}")
            currLoss = 0
            currAcc = 0
            lossArr = [] # temp variable
            accArr = []
            for ind, (img, label) in enumerate(self.trainData):
                print(
                    f"this is iteration number {ind + 1}/{len(self.trainData)} for epoch number {epoch + 1}/{args.epochs}")
                currX = img.squeeze()
                print(f"\n{currX.shape}\n")
                if not useRows:
                    currX = currX.view(currX.shape[0], args.pixel_seq_size, 1)


                self.optimizer.zero_grad()
                currX.to(self.device)
                output, classed = NN(currX)
                lossClass = nn.CrossEntropyLoss().forward(input=classed.squeeze(), target=label)
                loss = nn.MSELoss().forward(output, currX)
                totalLoss = (loss + lossClass) / 2
                totalLoss.backward()
                self.optimizer.step()
                currLoss += totalLoss.item()
                acc = self.accuracy(classed, label)
                currAcc += acc
                lossArr.append(totalLoss.item())
                accArr.append(acc)
                self.plotLoss(accArr, "curr accuracy mnist")
                self.plotLoss(lossArr, "curr loss mnist")

                if ind % 200 == 0:
                    self.showOneImg(output[0], "reconstructed")
                    self.showOneImg(currX[0], "orig")
                #     with torch.no_grad():
                #         self.imshow(torchvision.utils.make_grid(currX.unsqueeze(1)), "original", useRows)
                #         self.imshow(torchvision.utils.make_grid(output.unsqueeze(1)), "recontructed", useRows)
            avgLoss = currLoss / len(self.trainData)
            avgACC = currAcc / len(self.trainData)
            accuracy.append(avgACC)
            trainLoss.append(avgLoss)

        if useRows and saveNet:
            torch.save(self.AEC.state_dict(), classifyDir)
            print(f"Finished training. Saving net at {classifyDir}")

        elif saveNet:
            torch.save(self.pixel_AEC.state_dict(), pixelDir)
            print(f"Finished training. Saving net at {pixelDir}")

        if not saveNet:
            print(f"Finished training. not Saving net")

        return trainLoss, accuracy

    def reconstruct(self, data):
        return self.AE(data.type(torch.FloatTensor))

    def reconstructClass(self, data, useRows):
        return self.AEC(data.type(torch.FloatTensor)) if useRows else self.pixel_AEC(
            data.type(torch.FloatTensor))  # maybe remove type

    def plotNN(self, saveNet = False):

        startLoss = time.perf_counter()
        loss = self.train(saveNet)
        dataIter = iter(self.testData)
        figure, labels = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.imshow(figure, cmap='gray')
        plt.show()

        reconed = self.reconstruct(torch.unsqueeze(figure, 0))
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
        print(
            f"layers - {args.num_of_layers}\nepochs - {args.epochs}\nbatch size - {args.batch_size}\nlearning rate - {args.lr}\noptimizer - {args.optimizer}\n")
        print(f"the loss calc took {(endLoss - startLoss) / 60} minutes")
        print(f"the reconstruct calc took {(endReconstruct - endLoss) / 60} minutes")
        print(f"overall it took {(endReconstruct - startLoss) / 60} minutes")

    def plotClassification(self, useRows, saveNet=False):
        startLoss = time.perf_counter()

        trainLoss, accuracy = self.trainClassification(useRows, saveNet)

        plt.figure()
        plt.title("Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.arange(self.epochs), trainLoss)
        plt.show()

        plt.figure()
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.arange(self.epochs), accuracy)
        plt.show()

        dataIter = iter(self.testData)
        figure, labels = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.imshow(figure, cmap='gray')
        plt.show()

        reconed, label = self.reconstructClass(torch.unsqueeze(figure, 0), useRows) if useRows else self.reconstructClass(figure.view(1, args.pixel_seq_size, 1), useRows)
        fixedLabel = np.argmax(label.squeeze().detach().numpy())
        print(fixedLabel)
        plt.title("Reconstructed")
        if not useRows:
            reconed = reconed.view(28, 28)
        reconed = reconed / 0.3081 + 0.1307
        plt.imshow(reconed.detach().squeeze().numpy(), cmap='gray')
        plt.show()



        endLoss = time.perf_counter()
        print(f"overall it took {(endLoss - startLoss) / 60} minutes")

    def accuracy(self, predict, labels):
        newPredict = np.argmax(predict.squeeze().detach().numpy(), axis=1)
        return 1 - np.mean(newPredict != labels.detach().numpy())

    def imshow(self, img, title, useRows):
        #  Amir's show image
        img = img / 0.3081 + 0.1307
        numg = img.numpy()
        plt.imshow(np.transpose(numg, (1, 2, 0)))
        plt.title(title)
        plt.show()

    def showOneImg(self, img, title):

        img = img / 0.3081 + 0.1307
        # reconed = img.view(28, 28)
        plt.imshow(img.squeeze().detach().numpy().reshape(28,28), cmap='gray')
        plt.title(title)
        plt.show()

    def plotLoss(self, loss, title):
        plt.figure()
        plt.plot(loss)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


saveNet = False
MnistAE().plotClassification(False)
