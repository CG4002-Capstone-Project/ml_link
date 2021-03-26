import collections
import warnings

import numpy as np
from joblib import load
from scipy import signal, stats

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    print("warning: torch cannot be found")

warnings.filterwarnings("ignore")


class PositionDetection:
    def __init__(self, verbose):
        self.skip_count = 0

        # constants
        self.upper_position_threshold = 0.2 * 100
        self.lower_position_threshold = 0.2 * 100
        self.verbose = verbose

    def preprocess(self, data):
        gy = data[-3:, 1]

        upper_mask = np.any(gy > self.upper_position_threshold)
        lower_mask = np.any(gy < -self.lower_position_threshold)

        return lower_mask, upper_mask

    def infer(self, data):
        lower_mask, upper_mask = self.preprocess(data)

        if upper_mask:
            return "right"
        if lower_mask:
            return "left"

        return None


class DanceDetection:
    def __init__(self, model, model_type, scaler, verbose):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.verbose = verbose
        self.activities = ["gun", "sidepump", "hair"]
        self.accuracy = -1

    def preprocess(self, inputs):
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
        return inputs

    def infer(self, inputs):
        inputs = self.preprocess(inputs)

        if self.model_type == "dnn":
            inputs = torch.tensor(inputs)  # convert to tensor
            outputs = self.model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            dance_move = self.activities[predicted]

            data = outputs.data
            accuracy = torch.max(data[0]) / torch.sum(torch.abs(data[0]))
            self.accuracy = accuracy.item() * 100

            if self.verbose:
                print(f"{dance_move} detected")
            return dance_move
        elif self.model_type == "fpga":
            return "fpga not implemented"
        else:
            raise Exception("model is not supported")


class Inference:
    def __init__(
        self, model, model_type, scaler, verbose, infer_dance=True, infer_position=False
    ):
        self.window_size = 50
        self.dance_data = collections.deque([], self.window_size)
        self.skip_count = 0

        self.position_detection = PositionDetection(verbose)
        self.dance_detection = DanceDetection(model, model_type, scaler, verbose)

        # constants
        self.dance_threshold = 1 * 100
        self.verbose = verbose
        self.infer_dance = infer_dance
        self.infer_position = infer_position

    def append_readings(self, gx, gy, gz, ax, ay, az):
        """
        appends readings to buffer
        """
        self.dance_data.append([gx, gy, gz, ax, ay, az])

    def check_is_ready(self):
        """
        skip between moves and positions for n skip_counts
        """
        if self.skip_count > 0:
            self.skip_count = self.skip_count - 1
            return False
        return True

    def check_is_moving(self):
        """
        returns true if dancer is moving
        """
        data = np.array(self.dance_data)
        if len(data) < 5:
            return False

        gx, gy, gz = data[-3:, 0], data[-3:, 1], data[-3:, 2]

        gx_mask = np.all(np.abs(gx) < self.dance_threshold)
        gy_mask = np.all(np.abs(gy) < self.dance_threshold)
        gz_mask = np.all(np.abs(gz) < self.dance_threshold)
        is_still = gx_mask and gy_mask and gz_mask

        return not is_still

    def infer_dancer_left_right(self):
        """
        infers dance left right
        """
        data = np.array(self.dance_data)
        if len(data) < 5:
            return None
        return self.position_detection.infer(data)

    def infer_dancer_moves(self):
        """
        infers dance moves
        """
        data = np.array(self.dance_data)
        if len(data) < self.window_size:
            return None
        return self.dance_detection.infer(data)


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
