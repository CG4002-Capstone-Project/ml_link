import numpy as np
import torch
import torch.nn as nn
from joblib import load

from intcomm import SERIAL_PORT, IntComm


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 32, 5)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(32, 20, 5)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.5)

        self.pool1 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()
        self.dp3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(420, 64)
        self.relu3 = nn.ReLU()
        self.dp4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dp2(x)

        x = self.pool1(x)
        x = self.flat1(x)
        x = self.dp3(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dp4(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model_path = "cnn_model.pth"
    scaler_path = "cnn_std_scaler.bin"

    print("Loading model")
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Loading scaler")
    scaler = load(scaler_path)

    intcomm = IntComm(SERIAL_PORT)
    data = []

    while True:
        new_data = intcomm.get_acc_gyr_data()
        if len(data) > 49:
            data = data[-49:]
        data.append(new_data)

        if len(data) == 50:
            inp = np.array([np.array(data).transpose()])
            num_instances, num_time_steps, num_features = inp.shape
            inp = np.reshape(inp, newshape=(-1, num_features))
            inp = scaler.transform(inp)
            inp = np.reshape(
                inp, newshape=(num_instances, num_time_steps, num_features)
            )
            inp = torch.tensor(inp)
            out = model(inp.float())
            _, i = torch.max(out.data, 1)

            moves = ["hair", "sidepump", "gun", "idle", "logout"]
            print(moves[i.item()])
