import numpy as np
from joblib import load
import time

try:
    import torch
    import torch.nn as nn
    from cnns import MCNN
except:
    print("Torch import failed")


# No. of samples to determine position
POSITION_WINDOW = 25

# No. of samples to determine dance move
DANCE_SAMPLES = 75

# No. of samples altogether for dance
# The ML model would slide along this window to get average move
DANCE_WINDOW = 75

if POSITION_WINDOW < 150:
    print("Position window has been set to low value for testing")
    time.sleep(1)

activities = [
    "dab",
    "elbowkick",
    "listen",
    "pointhigh",
    "hair",
    "gun",
    "sidepump",
    "wipetable",
    "logout",
    "idle"
]

class ML():
    def __init__(self, on_fpga, dance_scaler_path, pos_scaler_path, dance_model_path = "", pos_model_path = ""):
        self.on_fpga = on_fpga
        self.reset()
        self.load_scalers(dance_scaler_path)
        if not on_fpga:
            self.load_models(dance_model_path)

    def load_scalers(self, dance_scaler_path):
        self.dance_scaler = load(dance_scaler_path)

    def load_models(self, dance_model_path):
        if not self.on_fpga:
            dance_model = MCNN()
            dance_model.load_state_dict(torch.load(dance_model_path, map_location='cpu'))
            dance_model.eval()
            self.dance_model = dance_model

    def reset(self):
        self.data = [[], [], []] # data for 3 dancers
        self.preds = np.zeros(10)

    def write_data(self, dancer_id, data):
        self.data[dancer_id].append(data[1:])

    def scale_dance_data(self, samples):
        inp = np.array([np.array(samples).transpose()])
        num_instances, num_time_steps, num_features = inp.shape
        inp = np.reshape(inp, newshape=(-1, num_features))
        inp = self.dance_scaler.transform(inp)
        inp = np.reshape(inp, newshape=(num_instances, num_time_steps, num_features))
        return inp

    def pred_dance_move(self):
        for samples in self.data:
            if len(samples) >= POSITION_WINDOW + DANCE_WINDOW:
                dance_samples = samples[-DANCE_SAMPLES:]
                dance_samples = self.scale_dance_data(dance_samples)

                if not self.on_fpga:
                    inp = torch.tensor(dance_samples)
                    out = self.dance_model(inp.float())
                    self.preds = self.preds + out.detach().numpy()
                else:
                    pass # TODO FPGA PREDICTION HERE

        return activities[np.argmax(self.preds)]

    # used for testing
    def pred_intermediate_dance_move(self):
        for samples in self.data:
            if len(samples) >= POSITION_WINDOW + DANCE_SAMPLES:
                dance_samples = samples[-DANCE_SAMPLES:]
                dance_samples = self.scale_dance_data(dance_samples)

                if not self.on_fpga:
                    inp = torch.tensor(dance_samples)
                    out = self.dance_model(inp.float())
                    self.preds = self.preds + out.detach().numpy()
                else:
                    pass # TODO FPGA PREDICTION HERE

        return activities[np.argmax(self.preds)]

    def get_pred(self):
        mx_samples = max([len(x) for x in self.data])

        if mx_samples >= POSITION_WINDOW + DANCE_WINDOW + 10: # 10 is a small buffer to account for network variation
            dance_move = self.pred_dance_move()
            self.reset()
            return (dance_move, [1,2,3])
        elif mx_samples >= DANCE_SAMPLES:
            dance_move = self.pred_intermediate_dance_move()
            print("Intermediate prediction", dance_move)

        return None

