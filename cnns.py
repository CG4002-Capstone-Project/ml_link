import numpy as np
import torch.nn as nn
from scipy import signal, stats


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


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(126, 64)
        self.dp1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(64, 16)
        self.dp2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(16, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)

        x = self.fc2(x)
        x = self.dp2(x)

        x = self.fc3(x)
        return x


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(9, 32, 3)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.6)

        self.pool1 = nn.MaxPool1d(2)
        self.flat1 = nn.Flatten()
        self.dp2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(928, 256)
        self.relu2 = nn.ReLU()
        self.dp3 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.dp4 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dp1(x)

        x = self.pool1(x)
        x = self.flat1(x)
        x = self.dp2(x)

        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dp3(x)

        x = self.fc2(x)
        x = self.dp4(x)

        x = self.fc3(x)

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
