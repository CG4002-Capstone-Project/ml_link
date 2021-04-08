import torch.nn as nn


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, 5)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.3)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()
        self.dp2 = nn.Dropout(0.7)

        self.fc1 = nn.Linear(224, 64)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(64, 10)
        self.sm1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.pool1(x)
        x = self.pool2(x)
        x = self.flat1(x)
        x = self.dp2(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.sm1(x)

        return x


class PCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(18, 8, 5)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()
        self.dp2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(232, 64)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.pool1(x)
        x = self.pool2(x)
        x = self.flat1(x)
        x = self.dp2(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dp3(x)

        x = self.fc2(x)

        return x
