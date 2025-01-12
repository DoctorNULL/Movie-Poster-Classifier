import cv2
import torch.cuda

from datahandler import DataHandler
from model import Model
import torch.nn as nn
from pickle import load


maxLen = 50
classesSize = 25

dev = "cuda" if torch.cuda.is_available() else "cpu"

test = DataHandler("test")

model = torch.load("modelBackUP3.pt").to(dev)
cost = nn.BCEWithLogitsLoss()

file = open("labels.data", 'rb')
labels = list(load(file))
file.close()

print(labels)

print("Start Deploying...")

model.eval()

with torch.no_grad():
    inp = None
    y = None
    while inp != "exit":
        inp = input("Enter Number or Image Path : ")
        if str.isnumeric(inp):
            inp = int(inp)

            test.getRawImage(inp)

            gt = []

            for i,ele in enumerate(test.data[inp].labels):
                if bool(int(ele)):
                    gt.append(labels[i])

            print("Ground Truth : ", gt)

            x,y = test[inp]

            predicted = model(x.unsqueeze(0).to(dev))
            y = y.unsqueeze(0).to(dev)
            loss = cost(predicted, y)
            predicted = torch.sigmoid(predicted).round()

            correct = (predicted == y).sum().sum().item()

            mask = (predicted == 1).squeeze().nonzero(as_tuple=True)[0]

            res = []

            for i in mask:
                res.append(labels[i.item()])

            print(res)

            print(f"Deployment Batch Loss {loss}"
                  f" Accuracy {correct / classesSize * 100} Correct {correct}/{classesSize}")

        else:

            x = cv2.imread(inp)

            if x is None:
                print("No image in this path")
                continue

            cv2.imshow("Image", x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            x = cv2.resize(x, (250, 250))

            x = torch.tensor(x, dtype=torch.float)
            x = x / 255
            x = x.permute((2, 0, 1))

            x = torch.tensor(x).unsqueeze(0).to(dev)

            predicted = model(x)
            predicted = torch.sigmoid(predicted).round()

            mask = (predicted == 1).squeeze().nonzero(as_tuple=True)[0]

            res = []

            for i in mask:
                res.append(labels[i.item()])

            print("Predicted : ", res)

