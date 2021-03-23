import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from inference import Inference


def load_data(file_name="data.csv"):
    df = pd.read_csv(file_name, sep=",", header=None)
    df.columns = ["yaw", "pitch", "roll", "accx", "accy", "accz"]
    df["accx"] = df["accx"] / 8192
    df["accy"] = df["accy"] / 8192
    df["accz"] = df["accz"] / 8192
    df["timestamp"] = [i for i in range(len(df))]
    print(df.head())
    print(df.describe())
    return df


def visualize_data(df):
    sns.set()

    fig, ax = plt.subplots(1, 2)

    sns.scatterplot(data=df, x="timestamp", y="accx", label="accx", ax=ax[0])
    sns.scatterplot(data=df, x="timestamp", y="accy", label="accy", ax=ax[0])
    sns.scatterplot(data=df, x="timestamp", y="accz", label="accz", ax=ax[0])
    ax[0].set(xlabel="readings", ylabel="acceleration")
    ax[0].legend()

    sns.scatterplot(data=df, x="timestamp", y="yaw", label="yaw", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="pitch", label="pitch", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="roll", label="roll", ax=ax[1])
    ax[1].set(xlabel="readings", ylabel="angle")
    ax[1].legend()

    plt.show()


def inference_data(df, verbose):
    inference = Inference(verbose)

    for timestamp in range(df.shape[0]):
        data = df.iloc[timestamp]
        yaw, pitch, roll, accx, accy, accz = (
            data.yaw,
            data.pitch,
            data.roll,
            data.accx,
            data.accy,
            data.accz,
        )
        inference.append_readings(yaw, pitch, roll, accx, accy, accz)
        result = inference.infer()
        if result:
            print(f"timestamp: {timestamp} result: {result}")


def main():
    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument("--verbose", default=True, help="verbose", type=bool)
    parser.add_argument("--visualize", default=True, help="svc or dnn model")

    args = parser.parse_args()
    verbose = args.verbose
    visualize = args.visualize

    print("verbose:", verbose)
    print("visualize:", visualize)

    df = load_data()

    if verbose:
        visualize_data(df)

    inference_data(df, verbose)


if __name__ == "__main__":
    main()
