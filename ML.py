import numpy as np
from joblib import load

from models import DNN, extract_raw_data_features

TRANSITION_WINDOW = 50

# No. of samples to determine position
POSITION_WINDOW = 40

# No. of samples to determine dance move
DANCE_SAMPLES = 60

# No. of samples altogether for dance
DANCE_WINDOW = 210

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
        self.is_first_prediction = True

    def load_scalers(self, dance_scaler_path):
        self.dance_scaler = load(dance_scaler_path)

    def load_models(self, dance_model_path):
        dance_model = DNN()
        dance_model.setup(dance_model_path)
        self.dance_model = dance_model

    def reset(self):
        self.dance_data = [[], [], []]  # data for 3 dancers
        self.position_data = [[], [], []]

    def write_data(self, dancer_id, data, data_type):
        if data_type == 0:
            self.dance_data[dancer_id].append(data)
        else:
            self.position_data[dancer_id].append(data)

    def scale_dance_data(self, samples):
        samples = np.array(samples)
        inputs = np.array(
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
        inputs = extract_raw_data_features(inputs)
        inputs = self.dance_scaler.transform(inputs)
        return inputs

    def pred_dance_move(self):
        inputs = [
            self.scale_dance_data(np.array(data)[-DANCE_SAMPLES:])
            for data in self.dance_data
            if len(data) > TRANSITION_WINDOW + POSITION_WINDOW + DANCE_WINDOW // 2
        ]
        inputs = np.array(inputs)
        outputs = self.dance_model(inputs)
        predictions = [activities[np.argmax(output)] for output in outputs]
        return predictions

    def pred_position(self):
        pos = ["S", "S", "S"]  # S - still, L - left, R - right

        for i in range(3):
            sample = np.array(self.position_data[i])

            if self.is_first_prediction:
                if sample.shape[0] < POSITION_WINDOW:
                    continue

                pitchs = sample[:, 1]
            else:
                if sample.shape[0] < TRANSITION_WINDOW + POSITION_WINDOW:
                    continue

                pitchs = sample[TRANSITION_WINDOW:, 1]

            # indices of roll less than -50 (right) and greater than 50 (left)
            right_pitchs_idxs, left_pitchs_idxs = (
                np.where((pitchs < -30))[0],
                np.where((pitchs > 30))[0],
            )
            # register a turn if more than 3 points are above threshold
            if left_pitchs_idxs.shape[0] >= 5 or right_pitchs_idxs.shape[0] >= 5:
                if right_pitchs_idxs.shape[0] == 0:
                    pos[i] = "R"
                elif left_pitchs_idxs.shape[0] == 0:
                    pos[i] = "L"
                else:
                    left_idx_mean = np.mean(left_pitchs_idxs)
                    right_idx_mean = np.mean(right_pitchs_idxs)
                    pos[i] = "R" if left_idx_mean > right_idx_mean else "L"
        return pos

    def get_pred(self):
        mx_samples = max([len(x) for x in self.dance_data])

        if (
            mx_samples >= POSITION_WINDOW + DANCE_WINDOW + TRANSITION_WINDOW + 25
        ):  # 10 is a small buffer to account for network variation
            dance_moves = self.pred_dance_move()
            pos = self.pred_position()
            sync_delay = self.pred_sync_delay()
            self.reset()
            return dance_moves, pos, sync_delay
        return None

    def pred_sync_delay(self):
        idxs = [self.get_start_index(self.dance_data[i]) for i in range(3)]
        idxs = [idx for idx in idxs if idx is not None]
        sync_delay = np.max(idxs) - np.min(idxs)
        return -1 if sync_delay == 0 else sync_delay / 25

    def get_start_index(self, dance_data):
        n_data = len(dance_data)
        if n_data < TRANSITION_WINDOW + POSITION_WINDOW + DANCE_WINDOW:
            return None
        pitchs = np.array(dance_data)[:, 1]
        pitchs = np.abs(pitchs[TRANSITION_WINDOW + POSITION_WINDOW :])
        idxs = np.where(pitchs > 30)[0]
        return pitchs.shape[0] if idxs.shape[0] == 0 else np.min(idxs)
