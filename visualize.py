import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(file_name):
    df = pd.read_csv(file_name, sep=",")
    df["ax"] = df["ax"] / 8192
    df["ay"] = df["ay"] / 8192
    df["az"] = df["az"] / 8192
    df["yaw"] = df["yaw"]
    df["pitch"] = df["pitch"]
    df["roll"] = df["roll"]
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

    sns.scatterplot(data=df, x="timestamp", y="yaw", label="yaw", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="pitch", label="pitch", ax=ax[1])
    sns.scatterplot(data=df, x="timestamp", y="roll", label="roll", ax=ax[1])
    ax[1].set(xlabel="readings", ylabel="angles")
    ax[1].legend()

    plt.show()


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

    df = load_data(visualize)
    visualize_data(df)


if __name__ == "__main__":
    main()
