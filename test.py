import torch.cuda

from datahandler import DataHandler
from torch.utils.data import DataLoader
import torch.nn as nn



batchSize = 50
classesSize = 25
weight = 0.9

dev = "cuda" if torch.cuda.is_available() else "cpu"

test = DataHandler("test")
testLoader = DataLoader(test, batchSize, True)

model = torch.load("modelBackUP4.pt").to(dev)

print("Start Testing...")

model.eval()

testCorrect = 0
testCost = 0
totalCorrectOnes = 0
totalOnes = 0

with torch.no_grad():
    for idx, (x, y) in enumerate(testLoader):
        predicted = model(x.to(dev))
        y_ = predicted.sigmoid().round()
        y = y.to(dev)
        w = y * weight + (1 - y) * (1- weight)
        cost = nn.BCEWithLogitsLoss(pos_weight=w)
        loss = cost(predicted, y)
        correct = ((torch.sigmoid(predicted).round()) == y).sum().sum().item()
        correctOnes = ((y_ == y) & (y_ == 1)).sum().sum().item()

        totalCorrectOnes += correctOnes
        totalOnes += y.sum().sum().sum().item()

        testCost += loss
        testCorrect += correct

        print(f"Testing Batch {idx * batchSize}/{len(test)} Loss {loss}"
              f" Accuracy {correct / (x.size(0) * classesSize) * 100} Correct {correct}/{batchSize * classesSize}"
              f" Correct Ones {correctOnes}/{y.sum().sum().sum().item()}"
              f" Ones Accuracy {correctOnes / y.sum().sum().sum().item() * 100}")

print(f"Testing Cost {testCost}"
      f" Testing Accuracy {testCorrect / (len(test) * classesSize) * 100} "
      f" Testing Correct {testCorrect}/{len(test) * classesSize}")
