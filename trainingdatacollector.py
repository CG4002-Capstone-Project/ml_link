import pandas as pd

from intcomm import IntComm

if __name__ == "__main__":

    # change this according to your serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0)
    data = []
    print("Start")
    try:
        while True:
            point = intcomm.get_acc_gyr_data()
            print("data is...")
            print(point)
            data.append(point)
    except KeyboardInterrupt:
        print("terminating program")
    except Exception:
        print("an error occured")

    df = pd.DataFrame(data)
    df.columns = ["gx", "gy", "gz", "ax", "ay", "az"]
    df.to_csv("green_left.csv", sep=",")
