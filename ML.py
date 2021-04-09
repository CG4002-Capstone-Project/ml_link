import os
import random

import numpy as np
from joblib import load

from models import DNN, extract_raw_data_features

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
    "hair",
    "listen",
    "sidepump",
    "dab",
    "wipetable",
    "gun",
    "elbowkick",
    "pointhigh",
    "logout",
]


class ML:
    def __init__(
        self, dance_scaler_path, dance_model_path,
    ):
        self.reset()
        self.load_scalers(dance_scaler_path)
        self.load_models(dance_model_path)
        self.set_pos([1, 2, 3])

    def load_scalers(self, dance_scaler_path):
        self.dance_scaler = load(dance_scaler_path)

    def load_models(self, dance_model_path):
        dance_model = DNN()
        dance_model.setup(dance_model_path)
        self.dance_model = dance_model

    def reset(self):
        self.data = [[], [], []]  # data for 3 dancers
        self.preds = np.zeros(9)

    def set_pos(self, p):
        self.pos = p

    def write_data(self, dancer_id, data):
        self.data[dancer_id].append(data)
        self.update_dance_pred(self.data[dancer_id])

    def scale_dance_data(self, samples):
        samples = np.array(samples)
        inp = np.array(
            [
                [
                    samples[:, 0],
                    samples[:, 1],
                    samples[:, 2],
                    samples[:, 3],
                    samples[:, 4],
                    samples[:, 5],
                    samples[:, 6],
                    samples[:, 7],
                    samples[:, 8],
                ]
            ]
        )
        inp = extract_raw_data_features(inp)
        inp = self.dance_scaler.transform(inp)
        return inp

    def update_dance_pred(self, samples):
        if len(samples) >= POSITION_WINDOW + DANCE_SAMPLES:
            dance_samples = samples[-DANCE_SAMPLES:]
            dance_samples = self.scale_dance_data(dance_samples)
            out = self.dance_model(dance_samples)
            self.preds = self.preds + out
            if "DEBUG" in os.environ:
                print("Intermediate prediction", self.pred_dance_move())

    def pred_dance_move(self):
        return activities[np.argmax(self.preds)]

    def pred_position(self):
        pos = ["S", "S", "S"]  # S - still, L - left, R - right

        for i in range(3):
            sample = np.array(self.data[i])
            if sample.shape[0] < TRANSITION_WINDOW + POSITION_WINDOW:
                continue

            gxs = sample[TRANSITION_WINDOW : TRANSITION_WINDOW + POSITION_WINDOW, 3]

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
            dance_move = self.pred_dance_move()
            # dance_move = None
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
