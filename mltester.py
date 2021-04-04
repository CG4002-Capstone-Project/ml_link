import time
import torch

from inference import Inference, load_model
from intcomm import IntComm
from ML import ML

if __name__ == "__main__":
    dance_model_path = "dance_model.pth"
    dance_scaler_path = "dance_scaler.bin"
    pos_model_path = "pos_model.pth"
    pos_scaler_path = "pos_scaler.bin"

    ml = ML(
        on_fpga = False,
        dance_scaler_path = dance_scaler_path,
        dance_model_path = dance_model_path,
        pos_scaler_path = pos_scaler_path,
        pos_model_path = pos_model_path
    )

    # change this according to your serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0)

    while True:
        line = intcomm.get_line()

        yaw, pitch, roll, accx, accy, accz, emg = line[1:].split(",")
        yaw, pitch, roll, accx, accy, accz = (
            float(yaw),
            float(pitch),
            float(roll),
            float(accx),
            float(accy),
            float(accz),
        )

        ml.write_data(0, [yaw, pitch, roll, accx, accy, accz])

        pred = ml.get_pred()

        if pred is not None:
            print("Prediction", pred)

