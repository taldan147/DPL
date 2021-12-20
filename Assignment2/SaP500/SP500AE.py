import math
import os

import torch
from torch.utils.data import DataLoader

import LSTM_AE as AE
import SP500Predictor as predAE
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import pandas as pd

parser = argparse.ArgumentParser(description="Arguments of Toy AE")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--epochs', type=int, default=200, help="number of epochs")
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


def parseData():
    stocks = pd.read_csv(dataPath)
    return stocks.sort_values(by='date')

    #splits the raw data by symbol and keep the close
def splitData(stocks, numGroups):
    stocks = stocks[["symbol", "close"]]
    stocksGroups = stocks.groupby('symbol')
    stocksNames = stocksGroups['symbol'].apply(lambda x: pd.Series(x.values)).unstack()
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


def splitDataByName(stocks, numGroups):
    stocks = stocks[["symbol", "close"]]
    stocksGroups = stocks.groupby('symbol')
    data = stocksGroups['close'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)
    trainInd, testInd = createRandomIndices(len(data.values), 0.8)
    trainList = data.values[trainInd]
    testList = data.values[testInd]
    trainData = np.row_stack(np.asarray(np.array_split(trainList, 19, axis=1)))
    trainTensor = toNormal(torch.FloatTensor(trainData))
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
        self.AEPred = predAE.LSTMAE(args.input_size, args.num_of_layers, args.seq_size-1, args.hidden_size, args.dropout,
                            args.output_size)
        self.optimizerPred = torch.optim.Adam(self.AEPred.parameters(), args.lr) if (
                    args.optimizer == "Adam") else torch.optim.SGD(self.AEPred.parameters(), lr=args.lr)

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
                    self.plotSignal(currX[0], f"Reconstructed\nBatch {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
            lossArr.append(np.mean(np.asarray(currLoss)))
            self.plotLoss(lossArr, "Stocks Temp Loss")

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

        finalData = validateData.unsqueeze(2)
        return nn.MSELoss().forward(self.AE(finalData), finalData).detach().numpy()

    def trainPredict(self, trainLoader, validateData, saveNet=False, savePlt=False):
        self.AEPred.to(self.device)
        print("Starting Train For Prediction")
        if saveNet:
            print("Will save net!")
        lossArr = []
        lossReconArr = []
        lossPredArr = []
        for epoch in range(self.epochs):
            print(f"this is epoch number {epoch + 1}")
            currLoss = []
            currLossRecon = []
            currLossPred = []
            for ind, tensor in enumerate(trainLoader):
                print(f"this is iteration number {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
                currX = tensor.unsqueeze(2)[:,: -1]
                currY = tensor.unsqueeze(2)[:,1 :]

                self.optimizerPred.zero_grad()
                currX.to(self.device)
                output, pred = self.AEPred(currX)
                lossRecon = nn.MSELoss().forward(output, currX)
                lossPred = nn.MSELoss().forward(pred.unsqueeze(2), currY)
                loss = lossPred + lossRecon
                loss.backward()
                self.optimizerPred.step()
                currLoss.append(loss.item())
                currLossRecon.append(lossRecon.item())
                currLossPred.append(lossPred.item())
                if ind % 500 == 0:
                    self.plotSignal(currX[0], f"Reconstructed\nBatch {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
            lossArr.append(np.mean(np.asarray(currLoss)))
            lossReconArr.append(np.mean(np.asarray(currLossRecon)))
            lossPredArr.append(np.mean(np.asarray(currLossPred)))
            self.plotPred(lossArr, lossReconArr, lossPredArr, savePlt)          #3.2

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

    def testPredict(self, dataLoader, savePlt=False):
        predKeeper = []
        loss = []
        interval = dataLoader.shape[1] - math.floor(dataLoader.shape[1]/2)
        for i in range(math.floor(dataLoader.shape[1]/2)):
            currX = dataLoader[:, i: i+interval]
            output, predict = self.AEPred(currX)
            predKeeper.append(predict[:, -1])
            currLoss = nn.MSELoss().forward(input=predict[:,-1], target=dataLoader[:, i+interval].squeeze())
            loss.append(currLoss.item())
        return predKeeper, np.mean(np.asarray(loss))



    # splits the data for the CrossValidate
    def prepareDataCrossValidate(self, trainTensor, ind):
        currTrain = trainTensor.copy()
        currValidate = currTrain.pop(ind)
        currTrain = torch.stack(currTrain)
        currTrain = torch.flatten(currTrain, 0, 1)
        return currTrain, currValidate

    def reconstruct(self, data):
        data.to(self.device)
        return self.AE.forward(data)

    def plotSignal(self, signal, title):
        signal = signal.squeeze()
        plt.title("Original")
        plt.plot(signal)
        plt.show()

        reconstructed = self.reconstruct(signal.unsqueeze(0).unsqueeze(2))
        plt.title(title)
        plt.plot(reconstructed.squeeze().detach().numpy())
        plt.savefig(f"Plots/ReconstructSignal.png")
        plt.show()

    def plotCrossVal(self, testData, savePlt=False):
        dataIter = iter(testData)
        figure = dataIter.next()
        figure = figure.squeeze()
        plt.title("Original")
        plt.plot(figure)
        plt.xlabel("Date")
        plt.ylabel("Closing Rate")
        if savePlt:
            plt.savefig(f"Plots/OriginalCrossVal.png")
        plt.show()

        reconstructed = self.reconstruct(figure.unsqueeze(0).unsqueeze(2))
        plt.title("Reconstructed")
        plt.plot(reconstructed.squeeze().detach().numpy())
        plt.xlabel("Date")
        plt.ylabel("Closing Rate")
        if savePlt:
            plt.savefig(f"Plots/ReconstructCrossVal.png")
        plt.show()

    def plotLoss(self, loss, title, savePlt=False):
        plt.figure()
        plt.plot(loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        if savePlt:
            plt.savefig(f"Plots/{title}.png")
        plt.show()

    def plotPred(self, totalLoss, reconLoss, predLoss, savePlt=False):
        self.plotLoss(totalLoss, "Total Loss", savePlt)
        self.plotLoss(reconLoss, "Reconstruction Loss", savePlt)
        self.plotLoss(predLoss, "Prediction Loss", savePlt)

    def runPrediction(self, savePlt=False):
        startTime = time.perf_counter()

        data, test = splitData(parseData(), 1)
        self.trainPredict(DataLoader(data[0], args.batch_size, drop_last=True), test, False)
        multiPredKeeper, multiLoss = self.testPredict(test.unsqueeze(2))
        reconed = self.reconstruct(test.unsqueeze(2)).squeeze()
        halfMark = test.shape[1] - math.floor(test.shape[1]/2)

        reconed = reconed[:, halfMark:]

        multiPredKeeper = torch.stack(multiPredKeeper, dim=1)

        endTime = time.perf_counter()

        print(f"overall time is {(endTime - startTime) / 60} minutes")

        plt.figure()
        plt.title("Reconstructed vs Multi Predicted")
        plt.plot(reconed[0].detach().numpy(), label="Reconstructed", color="blue")
        plt.plot(multiPredKeeper[0].detach().numpy(), label="Multi Predicted", color="tomato")
        plt.xlabel("Time")
        plt.ylabel("Closing Rate")
        plt.legend()
        if savePlt:
            plt.savefig(f"Plots/multiPredict.png")
        plt.show()








def plotGoogleAmazon(savePlt=False):
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
    if savePlt:
        plt.savefig(f"Plots/GOOGLAMZN.png")
    plt.show()

def crossValidate(data, k, savePlt=False):
    trainTensor, testTensor = splitData(data, k)
    lossArr = []
    startTime = time.perf_counter()
    endIter = 0
    for ind in range(k):
        print(f"Starting the {ind+1} validation set")
        sp500 = SP500AE()
        startIter = time.perf_counter()
        currTrain, currValidate = sp500.prepareDataCrossValidate(trainTensor, ind)
        trainLoader = DataLoader(currTrain, args.batch_size, drop_last=True)
        lossArr.append(sp500.train(trainLoader, currValidate))
        endIter = time.perf_counter()
        print(f"the {ind+1} validation took {(endIter - startIter)/60} mintues")
    bestArg = np.argmin(np.asarray(lossArr))
    bestTrain, _ = sp500.prepareDataCrossValidate(trainTensor, bestArg)
    print(f"Starting full train")
    bestLoss = SP500AE().train(DataLoader(bestTrain, args.batch_size, drop_last=True), testTensor)
    endTime = time.perf_counter()
    print(f"the best loss we got was {bestLoss}")
    print(f"training on the chosen part took {(endTime - endIter)/60} minutes")
    print(f"overall time is {(endTime - startTime)/60} minutes")
    sp500.plotCrossVal(DataLoader(testTensor, 1, drop_last=True), savePlt)


# crossValidate(parseData(),4)
# plotGoogleAmazon()
SP500AE().runPrediction()

#TODO: change date to time!
