import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from inference import Inference, load_model


def load_data(file_name, normalized=True):
    df = pd.read_csv(file_name, sep=",")
    if normalized:
        df["ax"] = df["ax"] / 8192
        df["ay"] = df["ay"] / 8192
        df["az"] = df["az"] / 8192
        df["gx"] = df["gx"] / 100
        df["gy"] = df["gy"] / 100
        df["gz"] = df["gz"] / 100
    df["timestamp"] = [i for i in range(len(df))]
    print(df.head())
    print(df.describe())
    return df


def visualize_data(df):
    sns.set()

    fig, ax = plt.subplots(1, 2)

    sns.scatterplot(data=df, x="timestamp", y="ax", label="ax", ax=ax[0])
    sns.scatterplot(data=df, x="timestamp", y="ay", label="ay", ax=ax[0])
    sns.scatterplot(data=df, x="timestamp", y="az", label="az", ax=ax[0])
    ax[0].set(xlabel="readings", ylabel="acceleration")
    ax[0].legend()

    sns.scatterplot(data=df, x="timestamp", y="gx", label="gx", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="gy", label="gy", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="gz", label="gz", ax=ax[1])
    ax[1].set(xlabel="readings", ylabel="gyroscope")
    ax[1].legend()

    plt.show()


def inference_data(df, verbose):
    model_path = "/home/nwjbrandon/models/dnn_model.pth"
    scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
    model_type = "dnn"
    model, scaler = load_model(model_type, model_path, scaler_path)

    inference = Inference(model, model_type, scaler, verbose)

    for timestamp in range(df.shape[0]):
        data = df.iloc[timestamp]
        gx, gy, gz, ax, ay, az = (
            data.gx,
            data.gy,
            data.gz,
            data.ax,
            data.ay,
            data.az,
        )
        inference.append_readings(gx, gy, gz, ax, ay, az)
        result = inference.infer()
        if result:
            print(f"timestamp: {timestamp} result: {result}")


def main():
    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument("--verbose", default=False, help="verbose", type=bool)
    parser.add_argument(
        "--visualize", default=False, help="file name",
    )

    args = parser.parse_args()
    verbose = args.verbose
    visualize = args.visualize

    print("verbose:", verbose)
    print("visualize:", visualize)

    if visualize:
        df = load_data(visualize)
        visualize_data(df)

    df = load_data(visualize, normalized=False)
    inference_data(df, verbose)


if __name__ == "__main__":
    main()
