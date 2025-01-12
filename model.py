import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, numClasses:int, dropout: float = 0.3):
        super().__init__()

        self.cnn1 = nn.Conv2d(3, 64, 3)
        self.cnn2 = nn.Conv2d(64, 64, 3)
        self.maxPool = nn.MaxPool2d(3, 2)

        self.cnn3 = nn.Conv2d(64, 128, 3)
        self.cnn4 = nn.Conv2d(128, 128, 3)

        self.cnn5 = nn.Conv2d(128, 256, 3)
        self.cnn6 = nn.Conv2d(256, 256, 3)

        self.cnn7 = nn.Conv2d(256, 512, 3)
        self.cnn8 = nn.Conv2d(512, 512, 3)
        self.cnn9 = nn.Conv2d(512, 512, 3)

        self.linear1 = nn.Linear(4608, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, numClasses)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Identity()
        self.cnnActivation = nn.Identity()

    def forward(self, x: torch.Tensor, ApplyLogit = False) -> torch.Tensor:
        #(Batch, 3, 250, 250)

        y = self.cnnActivation(self.cnn1(x))
        #(Batch, 64, 248, 248)

        y = self.cnnActivation(self.cnn2(y))
        #(Batch, 64, 246, 246)

        y = self.maxPool(y)
        # (Batch, 64, 122, 122)

        y = self.cnnActivation(self.cnn3(y))
        #(Batch, 128, 120, 120)

        y = self.cnnActivation(self.cnn4(y))
        #(Batch, 128, 118, 118)

        y = self.maxPool(y)
        # (Batch, 128, 58, 58)

        y = self.cnnActivation(self.cnn5(y))
        # (Batch, 256, 56, 56)

        y = self.cnnActivation(self.cnn6(y))
        # (Batch, 256, 54, 54)

        y = self.maxPool(y)
        # (Batch, 256, 26, 26)

        y = self.cnnActivation(self.cnn7(y))
        # (Batch, 512, 24, 24)

        y = self.cnnActivation(self.cnn8(y))
        # (Batch, 512, 22, 22)

        y = self.maxPool(y)
        # (Batch, 512, 10, 10)

        y = self.cnnActivation(self.cnn9(y))
        # (Batch, 512, 8, 8)

        y = self.maxPool(y)
        # (Batch, 512, 3, 3)

        y = y.flatten(1)
        #(Batch, 4608)

        y = self.activation(self.dropout(self.linear1(y)))
        #(Batch, 2048)

        y = self.activation(self.dropout(self.linear2(y)))
        #(Batch, 1024)

        y = self.activation(self.dropout(self.linear3(y)))
        # (Batch, 512)

        y = self.activation(self.dropout(self.linear4(y)))
        # (Batch, 128)

        y = self.activation(self.dropout(self.linear5(y)))
        # (Batch, 64)

        y = self.dropout(self.out(y))
        #(Batch, Classes)

        if ApplyLogit:
            y = nn.Sigmoid(y)

        return y
