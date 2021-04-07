import pandas as pd

from intcomm import IntComm

if __name__ == "__main__":

    # change this according to your serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0)
    all_data = []
    print("Start")
    try:
        while True:
            data = intcomm.get_line()
            print(data)
            if len(data) == 0 or data[0] != "#":
                print("Invalid data:", data)
                continue

            data = data[1:].split(",")
            if len(data) == 7:
                yaw, pitch, roll, accx, accy, accz, emg = data
                all_data.append([yaw, pitch, roll, accx, accy, accz])

    except KeyboardInterrupt:
        print("terminating program")
    except Exception:
        print("an error occured")

    df = pd.DataFrame(all_data)
    print(df.head())
    df.columns = ["yaw", "pitch", "roll", "ax", "ay", "az"]
    df.to_csv("test.csv", sep=",")
