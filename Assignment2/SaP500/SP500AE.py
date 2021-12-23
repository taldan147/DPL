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
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter

parser = argparse.ArgumentParser(description="Arguments of Toy AE")
parser.add_argument('--batch_size', type=int, default=8, help="batch size")
parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer to use")
parser.add_argument('--hidden_size', type=int, default=40, help="lstm hidden size")
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
meansTrain = []
stdsTrain = []
meansTest = []
stdsTest = []

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


def splitDataByName(stocks):
    stocks = stocks[["symbol", "close", "date"]]
    stocksGroups = stocks.groupby('symbol')
    data = stocksGroups['close'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)
    dates = stocksGroups['date'].apply(lambda x: pd.Series(x.values)).unstack()
    trainInd, testInd = createRandomIndices(len(data.values), 0.8)
    dataValues = np.asarray(data.values)
    trainList = dataValues[trainInd]
    testList = dataValues[testInd]
    trainData = np.asarray(np.array_split(trainList, 19, axis=1)).transpose((1,0,2))
    # trainData = np.asarray(toNormal(trainList))
    testData = np.asarray(np.array_split(testList, 19, axis=1)).transpose((1,0,2))
    # testData = np.asarray(toNormal(testList))
    trainTensor = torch.FloatTensor(toNormal(trainData, False))
    testTensor = torch.FloatTensor(toNormal(testData, True))

    # trainTensor = np.array_split(trainTensor, numGroups)

    return trainTensor, testTensor, np.asarray(dates[:1]).flatten()


def toNormal(data, isTestData):
    global meansTrain
    global stdsTrain
    global meansTest
    global stdsTest

    # data -= np.min(data,2, keepdims=True)
    # data /= np.max(data,2, keepdims=True)
    for i in range(len(data)):
        currMean = []
        currStd = []
        for j in range(len(data[i])):
            mean = np.mean(data[i][j])
            std = np.std(data[i][j])
            data[i][j] = (data[i][j] - mean) / std
            currMean.append(mean)
            currStd.append(std)
        if isTestData:
            meansTest.append(currMean)
            stdsTest.append(currStd)
        else:
            meansTrain.append(currMean)
            stdsTrain.append(currStd)
    return data

def fromNormal(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j] * stdsTest[i][j] + meansTest[i][j]
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
        model = self.AE.to(self.device)
        mse = nn.MSELoss().to(self.device)
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
                # currX = tensor.unsqueeze(2).to(self.device)
                currX = torch.flatten(tensor, 0,1).unsqueeze(2).to(self.device)
                self.optimizer.zero_grad()
                output = model.forward(currX)
                print(output.shape)
                loss = mse.forward(output, currX)
                loss.backward()
                self.optimizer.step()
                currLoss.append(loss.item())
                # if ind % 100 == 0:
                #     self.plotSignal(currX[0], f"Reconstructed\nBatch {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
            lossArr.append(np.mean(np.asarray(currLoss)))
            self.plotLoss(lossArr, "Stocks Temp Loss")

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")

        # finalData = torch.flatten(validateData, 0,1).unsqueeze(2).to(self.device)
        # finalData = validateData.unsqueeze(2).to(self.device)
        # return mse.forward(model.forward(finalData), finalData).detach().cpu().numpy()

    def trainPredict(self, trainLoader, validateData, saveNet=False, savePlt=False):
        model = self.AEPred.to(self.device)
        mse = nn.MSELoss().to(self.device)

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
                tensor = torch.flatten(tensor, 0,1).unsqueeze(2).to(self.device)
                currX = tensor[:,: -1].to(self.device)
                currY = tensor[:,1 :].to(self.device)

                self.optimizerPred.zero_grad()
                output, pred = model(currX)
                lossRecon = mse.forward(output, currX)
                lossPred = mse.forward(pred.unsqueeze(2), currY)
                loss = lossPred + lossRecon
                loss.backward()
                self.optimizerPred.step()
                currLoss.append(loss.item())
                currLossRecon.append(lossRecon.item())
                currLossPred.append(lossPred.item())
                currX.detach().cpu()
                currY.detach().cpu()
                # if ind % 500 == 0:
                #     self.plotSignal(currX[0], f"Reconstructed\nBatch {ind + 1}/{len(trainLoader)} for epoch number {epoch + 1}/{args.epochs}")
            lossArr.append(np.mean(np.asarray(currLoss)))
            lossReconArr.append(np.mean(np.asarray(currLossRecon)))
            lossPredArr.append(np.mean(np.asarray(currLossPred)))
        self.plotPred(lossArr, lossReconArr, lossPredArr, savePlt)          #3.2

        if saveNet:
            torch.save(self.AE.state_dict(), netDir)
            print(f"Finished training. Saving net at {netDir}")
        else:
            print(f"Finished training. Not saving net")


    # def testPredict(self, dataLoader, savePlt=False):
    #     predKeeper = []
    #     loss = []
    #     model = self.AEPred.to(self.device)
    #     mse = nn.MSELoss().to(self.device)
    #     interval = dataLoader.shape[1] - math.floor(dataLoader.shape[1]/2)
    #     for i in range(math.floor(dataLoader.shape[1]/2)):
    #         currX = dataLoader[:, i: i+interval].to(self.device)
    #         output, predict = model(currX)
    #         currX.detach()
    #         predKeeper.append(predict[:, -1])
    #         currLoss = mse.forward(input=predict[:,-1], target=dataLoader[:, i+interval].squeeze().to(self.device))
    #         loss.append(currLoss.item())
    #         currLoss.detach().cpu()
    #     return predKeeper, np.mean(np.asarray(loss))



    # splits the data for the CrossValidate
    def prepareDataCrossValidate(self, trainTensor, ind):
        currTrain = trainTensor.copy()
        currValidate = currTrain.pop(ind)
        currTrain = torch.stack(currTrain)
        currTrain = torch.flatten(currTrain, 0, 1)
        return currTrain, currValidate

    def reconstruct(self, data):
        return self.AE.to(self.device).forward(data.to(self.device))

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

    def plotCrossVal(self, testData, dates,savePlt=False):
        fig, axes = plt.subplots()
        axes.xaxis.set_major_locator(MaxNLocator(6))
        plt.xticks(rotation=20, ha='right')
        testData = fromNormal(testData)
        reconTest = self.AE(torch.flatten(torch.FloatTensor(testData), 0,1).unsqueeze(2).to(self.device)).view(testData.shape)
        reconTest = fromNormal(reconTest.detach().numpy())

        stock = testData[0]
        stock = stock.squeeze()
        plt.title("Original")
        plt.plot(dates, stock.flatten(), label='original')
        plt.xlabel("Date")
        plt.ylabel("Closing Rate")
        if savePlt:
            plt.savefig(f"Plots/OriginalCrossVal.png")

        plt.title("Original vs Reconstructed")

        plt.plot(dates, reconTest[0].flatten(), label='reconstructed')
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

        data, test, dates = splitDataByName(parseData())
        self.trainPredict(DataLoader(data, args.batch_size, drop_last=True), test, False)
        # multiPredKeeper, multiLoss = self.testPredict(test)
        _, oneStepPred = self.AEPred(torch.flatten(test, 0,1).unsqueeze(2).to(self.device))
        halfMark = test.shape[1] - math.floor(test.shape[1]/2)

        oneStepPred = oneStepPred[:, halfMark:]

        # multiPredKeeper = torch.stack(multiPredKeeper, dim=1)

        endTime = time.perf_counter()

        print(f"overall time is {(endTime - startTime) / 60} minutes")

        plt.figure()
        plt.title("One Step Predicted vs Multi Predicted")
        plt.plot(oneStepPred[0].detach().numpy(), label="Reconstructed", color="blue")
        # plt.plot(multiPredKeeper[0].detach().numpy(), label="Multi Predicted", color="tomato")
        plt.xlabel("Time")
        plt.ylabel("Closing Rate")
        plt.legend()
        if savePlt:
            plt.savefig(f"Plots/multiPredict.png")
        plt.show()



def plotGoogleAmazon(savePlt=False):
    stocks = parseData()
    stocks = stocks[["symbol", "high", "date"]]
    stocksGroups = stocks.groupby('symbol')
    data = stocksGroups['high'].apply(lambda x: pd.Series(x.values)).unstack()
    data.interpolate(inplace=True)
    dates = stocksGroups['date'].apply(lambda x: pd.Series(x.values)).unstack()
    dates = np.asarray(dates[:1]).flatten()
    fig, axes = plt.subplots()
    stocks = parseData()
    axes.xaxis.set_major_locator(MaxNLocator(6))
    plt.xticks(rotation=20, ha='right')
    google_amazon = stocks[stocks['symbol'].isin(["AMZN", "GOOGL"])]
    google_amazon = google_amazon.sort_values(by="date")
    amazon_daily_max = google_amazon[google_amazon.symbol == "AMZN"]['high']
    google_daily_max = google_amazon[google_amazon.symbol == "GOOGL"]['high']

    plt.plot(dates, amazon_daily_max, label='Amazon')
    plt.plot(dates, google_daily_max, label='Google')
    plt.legend()
    if savePlt:
        plt.savefig(f"Plots/GOOGLAMZN.png")
    plt.show()

def crossValidate(data, k, savePlt=False): #TODO make cross-validation work
    trainTensor, testTensor, dates = splitDataByName(data)
    trainTensor = np.array_split(trainTensor, k)
    lossArr = []
    startTime = time.perf_counter()
    endIter = 0
    # for ind in range(k):
    #     print(f"Starting the {ind+1} validation set")
    #     sp500 = SP500AE()
    #     startIter = time.perf_counter()
    #     currTrain, currValidate = sp500.prepareDataCrossValidate(trainTensor, ind)
    #     trainLoader = DataLoader(currTrain, args.batch_size, drop_last=True)
    #     lossArr.append(sp500.train(trainLoader, currValidate))
    #     endIter = time.perf_counter()
    #     print(f"the {ind+1} validation took {(endIter - startIter)/60} mintues")
    # bestArg = np.argmin(np.asarray(lossArr))
    # bestTrain, _ = sp500.prepareDataCrossValidate(trainTensor, bestArg)
    print(f"Starting full train")
    bestModer = SP500AE()
    bestLoss = bestModer.train(DataLoader(trainTensor, args.batch_size, drop_last=True), testTensor)
    endTime = time.perf_counter()
    print(f"the best loss we got was {bestLoss}")
    print(f"training on the chosen part took {(endTime - endIter)/60} minutes")
    print(f"overall time is {(endTime - startTime)/60} minutes")
    bestModer.plotCrossVal(testTensor,dates,  savePlt)
    # SP500AE().plotCrossVal(DataLoader(testTensor, 1, drop_last=True),dates,  savePlt)


# crossValidate(parseData(),2, savePlt=True)
plotGoogleAmazon()
# SP500AE().runPrediction(False)
#TODO: change date to time!
