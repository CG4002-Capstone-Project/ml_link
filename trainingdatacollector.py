import pandas as pd

from intcomm import SERIAL_PORT, IntComm

if __name__ == "__main__":

    intcomm = IntComm(SERIAL_PORT)
    data = []
    print("Start")
    try:
        while True:
            point = intcomm.get_acc_gyr_data()
            data.append(point)
    except KeyboardInterrupt:
        print("terminating program")
    except Exception:
        print("an error occured")

    df = pd.DataFrame(data)
    df.columns = ["gx", "gy", "gz", "ax", "ay", "az"]
    df.to_csv("data.csv", sep=",")
