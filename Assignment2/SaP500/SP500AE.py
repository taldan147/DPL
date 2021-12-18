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
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--epochs', type=int, default=300, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=100, help="lstm hidden size")
parser.add_argument('--num_of_layers', type=int, default=3, help="num of layers")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--input_size', type=int, default=1, help="size of an input")
parser.add_argument('--dropout', type=float, default=0, help="dropout ratio")
parser.add_argument('--seq_size', type=int, default=53, help="size of a seq")
parser.add_argument('--output_size', type=int, default=1, help="size of the output")
args = parser.parse_args()

dataPath = f"{os.getcwd()}/SP 500 Stock Prices 2014-2017.csv"
currDir = f"{os.getcwd()}/Sap500"
netDir = f"{currDir}/SavedNets/Net.pt"

minData = 0
maxData = 0

def parseData():
    stocks = pd.read_csv(dataPath)
    return stocks.sort_values(by='date')


def splitData(stocks, numGroups):
    stocks = stocks[["symbol", "close"]]
    stocksGroups = stocks.groupby('symbol')
    trainData = stocksGroups['close'].apply(lambda x: pd.Series(x.values)).unstack()
    trainData.interpolate(inplace=True)
    splittedStocksValues = np.row_stack(np.asarray(np.array_split(trainData.values, 19, axis=1)))
    trainInd, testInd = createRandomIndices(len(splittedStocksValues), 0.8)
    trainList = splittedStocksValues[trainInd]
    testList = splittedStocksValues[testInd]
    trainTensor = toNormal(torch.FloatTensor(trainList))
    testTensor = toNormal(torch.FloatTensor(testList))
    trainTensor = np.array_split(trainTensor, numGroups)

    return trainTensor, testTensor


def toNormal(data):
    minData = data.min(1, keepdim=True)[0]
    maxData = data.max(1, keepdim=True)[0]
    data -= minData
    data /= maxData
    return data


def createRandomIndices(n, trainSize):
    indices = np.random.permutation(n)
    train_size = int(n * trainSize)
    return indices[:train_size], indices[train_size:]


class SP500AE():
    def __init__(self):
        super(SP500AE, self).__init__()
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
        lossArr = []
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch + 1}")
            currLoss = []
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
                currLoss.append(loss.item())
                if ind % 100 == 0:
                    self.plotSignal(currX[0])
            lossArr.append(np.mean(np.asarray(currLoss)))
            self.plotLoss(lossArr)

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

        finalData = validateData.unsqueeze(2)
        return nn.MSELoss().forward(self.AE(finalData), finalData).detach().numpy()

    def crossValidate(self, data, k):
        trainTensor, testTensor = splitData(data, k)
        lossArr = []
        startTime = time.perf_counter()
        endIter = 0
        for ind in range(k):
            startIter = time.perf_counter()
            currTrain, currValidate = self.prepareData(trainTensor, ind)
            trainLoader = DataLoader(currTrain, args.batch_size, drop_last=True)
            lossArr.append(self.train(trainLoader, currValidate))
            endIter = time.perf_counter()
            print(f"the {ind+1} validation took {(endIter - startIter)/60} mintues")
        bestArg = np.argmin(np.asarray(lossArr))
        bestTrain, _ = self.prepareData(trainTensor, bestArg)
        bestLoss = self.train(DataLoader(bestTrain, args.batch_size, drop_last=True), testTensor)
        endTime = time.perf_counter()
        print(f"the best loss we got was {bestLoss}")
        print(f"training on the chosen part took {(endTime - endIter)/60} minutes")
        print(f"overall time is {(endTime - startTime)/60} minutes")
        self.plotCrossVal(DataLoader(testTensor, 1, drop_last=True))

    def prepareData(self, trainTensor, ind):
        currTrain = trainTensor.copy()
        currValidate = currTrain.pop(ind)
        currTrain = torch.stack(currTrain)
        currTrain = torch.flatten(currTrain, 0, 1)
        return currTrain, currValidate

    def reconstruct(self, data):
        data.to(self.device)
        return self.AE.forward(data)

    def plotSignal(self, signal):
        signal = signal.squeeze()
        plt.title("Original")
        plt.plot(signal)
        plt.show()

        reconstructed = self.reconstruct(signal.unsqueeze(0).unsqueeze(2))
        plt.title("reconstructed")
        plt.plot(reconstructed.squeeze().detach().numpy())
        plt.show()

    def plotCrossVal(self, testData):
        dataIter = iter(testData)
        figure = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.plot(figure)
        plt.show()

        reconstructed = self.reconstruct(figure.unsqueeze(0).unsqueeze(2))
        plt.title("reconstructed")
        plt.plot(reconstructed.squeeze().detach().numpy())
        plt.show()

    def plotLoss(self, loss):
        plt.figure()
        plt.plot(loss)
        plt.title("stocks temp loss")
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


SP500AE().crossValidate(parseData(), 2)
# plotGoogleAmazon()
