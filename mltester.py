from intcomm import IntComm
from ML import ML

if __name__ == "__main__":
    dance_model_path = "model_weights.json"
    dance_scaler_path = "dnn_std_scaler.bin"

    ml = ML(dance_scaler_path=dance_scaler_path, dance_model_path=dance_model_path,)

    # change this according to your serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0)

    while True:
        data = intcomm.get_line()
        if len(data) == 0 or data[0] != "#":
            print("Invalid data:", data)
            continue

        data = data[1:].split(",")
        if len(data) == 10:
            yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz, emg = data

            yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz = (
                float(yaw),
                float(pitch),
                float(roll),
                float(gyrox),
                float(gyroy),
                float(gyroz),
                float(accx),
                float(accy),
                float(accz),
            )
            ml.write_data(
                0, [yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz], 0
            )
            ml.write_data(
                0, [yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz], 1
            )
            pred = ml.get_pred()
            if pred is not None:
                print("Prediction", pred)
