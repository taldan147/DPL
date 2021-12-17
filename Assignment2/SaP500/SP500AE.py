import math
import os

import torch
from torch.utils.data import DataLoader

import LSTM_AE as AE
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import pandas as pd

parser = argparse.ArgumentParser(description="Arguments of Toy AE")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--epochs', type=int, default=5, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=200, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=3, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=1, help="size of an input")
parser.add_argument('--dropout', type=float, default=0, help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=1007, help="size of a seq")
parser.add_argument('--output_size', type=int, default=1, help="size of the output")
args = parser.parse_args()

dataPath = f"{os.getcwd()}/SP 500 Stock Prices 2014-2017.csv"
currDir = f"{os.getcwd()}/Sap500"
netDir = f"{currDir}/SavedNets/Net.pt"


def parseData():
    stocks = pd.read_csv(dataPath)
    print(stocks.keys())
    return stocks.sort_values(by='date')


def splitData(stocks, numGroups):
    stocks = stocks[["symbol", "close"]]
    stocksGroups = stocks.groupby('symbol')
    trainData = stocksGroups['close'].apply(lambda x: pd.Series(x.values)).unstack()
    trainData.interpolate(inplace=True)
    trainInd, testInd = createRandomIndices(len(trainData), 0.8)
    trainList = trainData.values[trainInd]
    testList = trainData.values[testInd]
    trainTensor = toNormal(torch.FloatTensor(trainList))
    testTensor = toNormal(torch.FloatTensor(testList))
    trainTensor = np.array_split(trainTensor, numGroups)
    testTensor = np.array_split(testTensor, numGroups)

    return trainTensor, testTensor


def toNormal(data):
    data -= data.min(1, keepdim=True)[0]
    data /= data.max(1, keepdim=True)[0]
    return data


def createRandomIndices(n, trainSize):
    indices = np.random.permutation(n)
    train_size = int(n * trainSize)
    return indices[:train_size], indices[train_size:]


class SP500AE():
    def __init__(self):
        super(SP500AE, self).__init__()
        # data = []
        # self.trainData = data[:int(0.6 * len(data)), :]
        # self.validateData = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
        self.epochs = args.epochs
        self.batchs = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = AE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size, args.hidden_size, args.dropout,
                            args.output_size)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), args.lr) if (
                    args.optimizer == "Adam") else torch.optim.SGD(self.AE.parameters(), lr=args.lr)

        print(f"using {self.device} as computing unit")

    def train(self, trainLoader, validateData, saveNet=False):
        self.AE.to(self.device)
        print("Starting Train")
        if saveNet:
            print("Will save net!")

        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch + 1}")
            for ind, tensor in enumerate(trainLoader):
                print(
                    f"this is iteration number {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
                currX = tensor.unsqueeze(2)
                self.optimizer.zero_grad()
                currX.to(self.device)
                output = self.AE.forward(currX)
                loss = nn.MSELoss().forward(output, currX)
                loss.backward()
                self.optimizer.step()

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

        return nn.MSELoss().forward(self.AE(validateData), validateData)

    def crossValidate(self, data, k):
        trainTensor, testTensor = splitData(data, k)
        lossArr = []
        startTime = time.perf_counter()
        for ind in range(k):
            startIter = time.perf_counter()
            currTrain, currValidate = self.prepareData(trainTensor, ind)
            trainLoader = DataLoader(currTrain, args.batch_size, drop_last=True)
            lossArr.append(self.train(trainLoader, currValidate))
            endIter = time.perf_counter()
            print(f"the {k+1} iteration took {(endIter - startIter)/60} mintues")
        bestArg = np.argmin(np.asarray(lossArr))
        bestTrain = self.prepareData(trainTensor, bestArg)
        bestLoss = self.train(DataLoader(bestTrain, args.batch_size, drop_last=True), testTensor)
        endTime = time.perf_counter()
        print(f"overall time is {(endTime - startTime)/60} minutes")
        self.plotCrossVal(DataLoader(testTensor, args.batch_size, drop_last=True))

    def prepareData(self, trainTensor, ind):
        currTrain = trainTensor.copy()
        currValidate = currTrain.pop(ind)
        currTrain = torch.stack(currTrain)
        currTrain = torch.flatten(currTrain, 0, 1)
        return currTrain, currValidate

    def reconstruct(self, data):
        data.to(self.device)
        return self.AE.forward(data)

    def plotCrossVal(self, testData):
        dataIter = iter(testData)
        figure = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.plot(figure)
        plt.show()

        reconstructed = self.reconstruct(figure)
        reconstructed.squeeze()
        plt.title("reconstructed")
        plt.plot(reconstructed)
        plt.show()



def plotGoogleAmazon():
    stocks = parseData()
    google_amazon = stocks[stocks['symbol'].isin(["AMZN", "GOOGL"])]
    google_amazon = google_amazon.sort_values(by="date")
    amazon_daily_max = google_amazon[google_amazon.symbol == "AMZN"]['high']
    google_daily_max = google_amazon[google_amazon.symbol == "GOOGL"]['high']
    amazon_daily_max.plot(x='date', y='high', title='amazon daily max', xlabel='Time', ylabel='Daily high',
                          label='Amazon')
    google_daily_max.plot(x='date', y='high', title='google daily max', xlabel='Time', ylabel='Daily high',
                          label='Google')
    plt.legend()
    plt.show()


SP500AE().crossValidate(parseData(), 4)
# plotGoogleAmazon()
