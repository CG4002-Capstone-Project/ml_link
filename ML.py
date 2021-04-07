import random
import time

import numpy as np
from joblib import load

try:
    import torch

    from cnns import MCNN, PCNN
except:
    print("Torch import failed")


# No. of samples to determine position
POSITION_WINDOW = 90

# No. of samples to determine dance move
DANCE_SAMPLES = 60

# No. of samples altogether for dance
# The ML model would slide along this window to get average move
DANCE_WINDOW = 180

if POSITION_WINDOW < 150:
    print("WARNING: Position window has been set to low value for testing")
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
    "idle",
]


class ML:
    def __init__(
        self,
        on_fpga,
        dance_scaler_path,
        pos_scaler_path,
        dance_model_path="",
        pos_model_path="",
    ):
        self.on_fpga = on_fpga
        self.reset()
        self.load_scalers(dance_scaler_path, pos_scaler_path)
        if not on_fpga:
            self.load_models(dance_model_path, pos_model_path)
        self.set_pos([1, 2, 3])

    def load_scalers(self, dance_scaler_path, pos_scaler_path):
        self.dance_scaler = load(dance_scaler_path)
        self.pos_scaler = load(pos_scaler_path)

    def load_models(self, dance_model_path, pos_model_path):
        if not self.on_fpga:
            dance_model = MCNN()
            dance_model.load_state_dict(
                torch.load(dance_model_path, map_location="cpu")
            )
            dance_model.eval()
            self.dance_model = dance_model

            pos_model = PCNN()
            pos_model.load_state_dict(torch.load(pos_model_path, map_location="cpu"))
            pos_model.eval()
            self.pos_model = pos_model

    def reset(self):
        self.data = [[], [], []]  # data for 3 dancers
        self.preds = np.zeros(10)

    def set_pos(self, p):
        self.pos = p

    def write_data(self, dancer_id, data):
        self.data[dancer_id].append(data)
        self.update_dance_pred(self.data[dancer_id])

    def scale_dance_data(self, samples):
        inp = np.array([np.array(samples).transpose()])
        num_instances, num_time_steps, num_features = inp.shape
        inp = np.reshape(inp, newshape=(-1, num_features))
        inp = self.dance_scaler.transform(inp)
        inp = np.reshape(inp, newshape=(num_instances, num_time_steps, num_features))
        return inp

    def scale_pos_data(self, samples):
        inp = np.array([np.array(samples).transpose()])
        num_instances, num_time_steps, num_features = inp.shape
        inp = np.reshape(inp, newshape=(-1, num_features))
        inp = self.pos_scaler.transform(inp)
        inp = np.reshape(inp, newshape=(num_instances, num_time_steps, num_features))
        return inp

    def update_dance_pred(self, samples):
        if len(samples) >= POSITION_WINDOW + DANCE_SAMPLES:
            dance_samples = samples[-DANCE_SAMPLES:]
            dance_samples = self.scale_dance_data(dance_samples)

            if not self.on_fpga:
                inp = torch.tensor(dance_samples)
                out = self.dance_model(inp.float())
                self.preds = self.preds + out.detach().numpy()
            else:
                pass  # TODO FPGA PREDICTION HERE

    def pred_dance_move(self):
        return activities[np.argmax(self.preds)]

    def pred_position(self):
        idle_point = [0] * 6  # Sentinel value for missing data

        samples = []  # need to insert POSITION_WINDOW x 18 data
        for i in range(POSITION_WINDOW):
            samples.append([])

        # add data in if available
        for x in self.pos:
            i = x - 1
            for j in range(POSITION_WINDOW):
                if len(self.data[i]) >= POSITION_WINDOW:
                    samples[j] += self.data[i][j]
                else:
                    samples[j] += idle_point

        result = 0
        samples = self.scale_pos_data(samples)

        if not self.on_fpga:
            inp = torch.tensor(samples)
            out = self.pos_model(inp.float())
            result = np.argmax(out.detach().numpy())
        else:
            pass  # TODO

        # Get permutation and map back
        perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        result = perms[result]

        # permute back
        # TODO VERIFY LOGIC
        result[0] = self.pos[result[0] - 1]
        result[1] = self.pos[result[1] - 1]
        result[2] = self.pos[result[2] - 1]

        return result

    def get_pred(self):
        mx_samples = max([len(x) for x in self.data])

        if (
            mx_samples >= POSITION_WINDOW + DANCE_WINDOW + 10
        ):  # 10 is a small buffer to account for network variation
            dance_move = self.pred_dance_move()
            pos = self.pred_position()
            sync_delay = self.pred_sync_delay()
            self.reset()
            return dance_move, pos, sync_delay
        elif mx_samples >= POSITION_WINDOW + DANCE_SAMPLES:
            dance_move = self.pred_dance_move()
        return None

    def get_start_index(self, dance_data):
        n_dance_data = len(dance_data)
        if n_dance_data < POSITION_WINDOW + DANCE_WINDOW:
            return None
        for idx in range(25, len(dance_data)):
            pitch = dance_data[idx][1]
            if abs(pitch) > 30:
                return idx
        return n_dance_data

    def pred_sync_delay(self):
        idxs = []
        for i in range(3):
            idx = self.get_start_index(self.data[i])
            if idx is not None:
                idxs.append(idx)
        print(idxs)
        sync_delay = max(idxs) - min(idxs)
        if sync_delay == 0:
            return random.random()
        return sync_delay / 25 * 1000
