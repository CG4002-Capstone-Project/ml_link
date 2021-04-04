import torch
import torch.nn as nn

class MCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 32, 8)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()

        self.fc1 = nn.Linear(320, 64)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(64, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.pool1(x)
        x = self.pool2(x)
        x = self.flat1(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dp3(x)

        x = self.fc2(x)
        x = self.sm(x)

        return x
