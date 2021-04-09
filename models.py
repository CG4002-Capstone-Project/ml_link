import json
import warnings

import numpy as np
from scipy import signal, stats

warnings.filterwarnings("ignore")


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
    normalized_spectrum = spectrum / np.sum(
        spectrum
    )  # like a probability mass function
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


def extract_raw_data_features(X):
    new_features = np.ones((X.shape[0], 126))
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


class FCLayer:
    def __init__(self, in_features, out_features, name="fclayer"):
        self.weights = np.zeros((out_features, in_features))
        self.bias = np.zeros(out_features)

    def forward(self, inputs):
        return inputs.dot(self.weights.T) + self.bias

    def __call__(self, inputs):
        return self.forward(inputs)

    def setup(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = np.array(bias)


class DNN:
    def __init__(self):
        self.fc1 = FCLayer(126, 64)
        self.fc2 = FCLayer(64, 16)
        self.fc3 = FCLayer(16, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)

    def setup(self, file_name="model_weights.json"):
        with open(file_name, "r") as f:
            model_weights = json.load(f)
            self.fc1.setup(model_weights["fc1"]["weight"], model_weights["fc1"]["bias"])
            self.fc2.setup(model_weights["fc2"]["weight"], model_weights["fc2"]["bias"])
            self.fc3.setup(model_weights["fc3"]["weight"], model_weights["fc3"]["bias"])
