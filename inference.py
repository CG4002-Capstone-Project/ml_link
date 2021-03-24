import collections
import warnings

import numpy as np
import torch
import torch.nn as nn
from joblib import load
from scipy import signal, stats

warnings.filterwarnings("ignore")


class PositionDetection:
    def __init__(self, verbose):
        self.position_mode = False
        self.position_direction = 0

        # constants
        self.position_threshold = 0.15
        self.verbose = verbose

    def get_mask(self, data):
        accx_mask = np.any(data[:, 3] > self.position_threshold)
        accy_mask = np.any(data[:, 4] > self.position_threshold)
        accz_mask = np.any(data[:, 5] > self.position_threshold)
        acc_right_mask = accx_mask or accy_mask or accz_mask

        accx_mask = np.any(data[:, 3] < -self.position_threshold)
        accy_mask = np.any(data[:, 4] < -self.position_threshold)
        accz_mask = np.any(data[:, 5] < -self.position_threshold)
        acc_left_mask = accx_mask or accy_mask or accz_mask

        return acc_left_mask, acc_right_mask

    def infer(self, data):
        acc_left_mask, acc_right_mask = self.get_mask(data)

        if self.position_mode:
            if self.position_direction == -1 and acc_right_mask:
                if self.verbose:
                    print("left move detected")
                self.position_mode = False
                self.position_direction = 0
                return True, "left"
            if self.position_direction == 1 and acc_left_mask:
                if self.verbose:
                    print("right move detected")
                self.position_mode = False
                self.position_direction = 0
                return True, "right"
        else:
            if acc_left_mask:
                self.position_direction = -1
                self.position_mode = True
            if acc_right_mask:
                self.position_direction = 1
                self.position_mode = True

        return False, 0


class DanceDetection:
    def __init__(self, model, model_type, scaler, verbose):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.verbose = verbose
        self.activities = ["gun", "sidepump", "hair"]

    def infer(self, inputs):
        inputs = np.array(
            [
                [
                    inputs[:, 0],
                    inputs[:, 1],
                    inputs[:, 2],
                    inputs[:, 3],
                    inputs[:, 4],
                    inputs[:, 5],
                ]
            ]
        )
        inputs = extract_raw_data_features(inputs)  # extract features
        inputs = scale_data(inputs, self.scaler)  # scale features
        if self.model_type == "dnn":
            inputs = torch.tensor(inputs)  # convert to tensor
            outputs = self.model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            dance_move = self.activities[predicted]
            if self.verbose:
                print(f"{dance_move} detected")
            return dance_move
        elif self.model_type == "fpga":
            return "NA"
        else:
            raise Exception("model is not supported")


class Inference:
    def __init__(self, model, model_type, scaler, verbose):
        self.idle_window_size = 3
        self.dance_window_size = 60
        self.total_window_size = 300
        self.idle_mode_data = collections.deque([], maxlen=self.idle_window_size)
        self.dance_data = collections.deque([], maxlen=self.total_window_size)
        self.skip_count = 0
        self.is_start = False
        self.counter = 0

        self.position_detection = PositionDetection(verbose)
        self.dance_detection = DanceDetection(model, model_type, scaler, verbose)

        # constants
        self.dance_threshold = 15
        self.verbose = verbose
        self.skip_interval = 30

    def append_readings(self, yaw, pitch, roll, accx, accy, accz):
        self.idle_mode_data.append([yaw, pitch, roll, accx, accy, accz])
        if self.is_start:
            self.dance_data.append([yaw, pitch, roll, accx, accy, accz])

    def get_mask(self, data):
        yaw_mask = np.any(data[:, 0] > self.dance_threshold)
        pitch_mask = np.any(data[:, 1] > self.dance_threshold)
        roll_mask = np.any(data[:, 2] > self.dance_threshold)
        angle_front_mask = yaw_mask or pitch_mask or roll_mask

        yaw_mask = np.any(data[:, 0] < -self.dance_threshold)
        pitch_mask = np.any(data[:, 1] < -self.dance_threshold)
        roll_mask = np.any(data[:, 2] < -self.dance_threshold)
        angle_back_mask = yaw_mask or pitch_mask or roll_mask

        return angle_front_mask or angle_back_mask

    def infer(self):
        if len(self.idle_mode_data) < self.idle_window_size or self.skip_count > 0:
            self.skip_count = self.skip_count - 1 if self.skip_count > 0 else 0
            return None

        data = np.array(self.idle_mode_data)

        # infer left right moves if dancer is in idle mode
        is_moving = self.get_mask(data)
        if not self.is_start:
            self.counter += 1
            if self.counter % 10 == 0:
                print("idling")
            if is_moving:
                self.is_start = True
                self.skip_count = self.skip_interval
            return None

        if not is_moving:
            is_move_detected, move = self.position_detection.infer(data)
            if is_move_detected:
                self.skip_count = self.skip_interval
                return move

        # infer dance moves if dancer is not in idle mode
        if len(self.dance_data) < self.total_window_size:
            return None
        data = np.array(self.dance_data)[-self.dance_window_size :]
        move = self.dance_detection.infer(data)
        self.skip_count = self.skip_interval
        return move

    def clear(self):
        self.idle_mode_data = collections.deque([], maxlen=self.idle_window_size)
        self.dance_data = collections.deque([], maxlen=self.total_window_size)


def scale_data(data, scaler, is_train=False):
    """
        data: inputs of shape (num_instances, num_features, num_time_steps)
        scaler: standard scalar to scale data
    """
    if is_train:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    return data


def compute_mean(data):
    return np.mean(data)


def compute_variance(data):
    return np.var(data)


def compute_median_absolute_deviation(data):
    return stats.median_absolute_deviation(data)


def compute_root_mean_square(data):
    def compose(*fs):
        def wrapped(x):
            for f in fs[::-1]:
                x = f(x)
            return x

        return wrapped

    rms = compose(np.sqrt, np.mean, np.square)
    return rms(data)


def compute_interquartile_range(data):
    return stats.iqr(data)


def compute_percentile_75(data):
    return np.percentile(data, 75)


def compute_kurtosis(data):
    return stats.kurtosis(data)


def compute_min_max(data):
    return np.max(data) - np.min(data)


def compute_signal_magnitude_area(data):
    return np.sum(data) / len(data)


def compute_zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum()


def compute_spectral_centroid(data):
    spectrum = np.abs(np.fft.rfft(data))
    normalized_spectrum = spectrum / np.sum(spectrum)
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
    return spectral_centroid


def compute_spectral_entropy(data):
    freqs, power_density = signal.welch(data)
    return stats.entropy(power_density)


def compute_spectral_energy(data):
    freqs, power_density = signal.welch(data)
    return np.sum(np.square(power_density))


def compute_principle_frequency(data):
    freqs, power_density = signal.welch(data)
    return freqs[np.argmax(np.square(power_density))]


def extract_raw_data_features_per_row(f_n):
    f1_mean = compute_mean(f_n)
    f1_var = compute_variance(f_n)
    f1_mad = compute_median_absolute_deviation(f_n)
    f1_rms = compute_root_mean_square(f_n)
    f1_iqr = compute_interquartile_range(f_n)
    f1_per75 = compute_percentile_75(f_n)
    f1_kurtosis = compute_kurtosis(f_n)
    f1_min_max = compute_min_max(f_n)
    f1_sma = compute_signal_magnitude_area(f_n)
    f1_zcr = compute_zero_crossing_rate(f_n)
    f1_sc = compute_spectral_centroid(f_n)
    f1_entropy = compute_spectral_entropy(f_n)
    f1_energy = compute_spectral_energy(f_n)
    f1_pfreq = compute_principle_frequency(f_n)
    return (
        f1_mean,
        f1_var,
        f1_mad,
        f1_rms,
        f1_iqr,
        f1_per75,
        f1_kurtosis,
        f1_min_max,
        f1_sma,
        f1_zcr,
        f1_sc,
        f1_entropy,
        f1_energy,
        f1_pfreq,
    )


def extract_raw_data_features(X, n_features=84):
    new_features = np.ones((X.shape[0], n_features))
    rows = X.shape[0]
    cols = X.shape[1]
    for row in range(rows):
        features = []
        for col in range(cols):
            f_n = X[row][col]
            feature = extract_raw_data_features_per_row(f_n)
            features.extend(feature)
        new_features[row] = np.array(features)
    return new_features


def load_model(model_type, model_path, scaler_path):
    scaler = load(scaler_path)
    if model_type == "dnn":

        class DNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(84, 64)
                self.dp1 = nn.Dropout(0.1)

                self.fc2 = nn.Linear(64, 16)
                self.dp2 = nn.Dropout(0.1)

                self.fc3 = nn.Linear(16, 3)

            def forward(self, x):
                x = self.fc1(x)
                x = self.dp1(x)

                x = self.fc2(x)
                x = self.dp2(x)

                x = self.fc3(x)
                return x

        model = DNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model, scaler

    elif model_type == "fpga":
        pass
    else:
        raise Exception("model is not supported")