import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def genData():
    randData = torch.rand(10000, 50, 1)
    for i in range(10000):
        j = random.randint(20, 30)
        for k in range(j-5, j+6, 1):
            randData[i][k] *= 0.1
    return randData


def exampleData(randData):
    example = random.sample(range(10000), 3)
    plt.figure()
    for i in example:
        plt.plot(randData[i])
    plt.title("Random Sample From Synthetic Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

