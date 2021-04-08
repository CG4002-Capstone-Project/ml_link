import os
import random

import numpy as np
from joblib import load

try:
    import torch

    from cnns import MCNN, PCNN
except:
    print("Torch import failed")


TRANSITION_WINDOW = 40

# No. of samples to determine position
POSITION_WINDOW = 50

# No. of samples to determine dance move
DANCE_SAMPLES = 60

# No. of samples altogether for dance
# The ML model would slide along this window to get average move
DANCE_WINDOW = 180

if POSITION_WINDOW < 150:
    print("WARNING: Position window has been set to low value for testing")

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
        # self.update_dance_pred(self.data[dancer_id])

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
                if "DEBUG" in os.environ:
                    print("Intermediate prediction", self.pred_dance_move())
            else:
                pass  # TODO FPGA PREDICTION HERE

    def pred_dance_move(self):
        return activities[np.argmax(self.preds)]

    def pred_position(self):
        pos = ["S", "S", "S"]  # S - still, L - left, R - right

        for i in range(3):
            sample = np.array(self.data[i])
            if sample.shape[0] < TRANSITION_WINDOW + POSITION_WINDOW:
                continue

            gxs = sample[TRANSITION_WINDOW : TRANSITION_WINDOW + POSITION_WINDOW, 3]
            print(gxs)

            # indices of roll less than -25 (right) and greater than 25 (left)
            right_gxs_idxs, left_gxs_idxs = (
                np.where((gxs < -50))[0],
                np.where((gxs > 50))[0],
            )

            right_gxs_count, left_gxs_count = (
                right_gxs_idxs.shape[0],
                left_gxs_idxs.shape[0],
            )

            # register a turn if more than 3 points are above threshold
            if left_gxs_count >= 3 or right_gxs_count >= 3:
                if left_gxs_count == 0:
                    pos[i] = "R"
                    continue
                if right_gxs_count == 0:
                    pos[i] = "L"
                    continue

                left_max, right_max = np.max(left_gxs_idxs), np.max(right_gxs_idxs)
                pos[i] = "L" if left_max < right_max else "R"

        return pos

    def get_pred(self):
        mx_samples = max([len(x) for x in self.data])

        if (
            mx_samples >= POSITION_WINDOW + DANCE_WINDOW + TRANSITION_WINDOW
        ):  # 10 is a small buffer to account for network variation
            # dance_move = self.pred_dance_move()
            dance_move = None
            pos = self.pred_position()
            sync_delay = self.pred_sync_delay()
            self.reset()
            return dance_move, pos, sync_delay
        return None

    def get_start_index(self, dance_data):
        n_samples = len(dance_data)
        if n_samples < TRANSITION_WINDOW + POSITION_WINDOW + DANCE_WINDOW:
            return None
        pitchs = np.array(dance_data)[:, 1]
        pitchs = np.abs(pitchs[TRANSITION_WINDOW + POSITION_WINDOW :])
        idxs = np.where(pitchs > 30)[0]
        return np.min(idxs) if idxs.shape[0] != 0 else pitchs.shape[0]

    def pred_sync_delay(self):
        idxs = [self.get_start_index(self.data[i]) for i in range(3)]
        idxs = [idx for idx in idxs if idx is not None]
        sync_delay = np.max(idxs) - np.min(idxs)
        return random.random() if sync_delay == 0 else sync_delay / 25 * 1000
