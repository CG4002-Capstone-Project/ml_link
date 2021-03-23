import time

from inference import Inference, load_model
from intcomm import SERIAL_PORT, IntComm

if __name__ == "__main__":
    model_path = "/home/nwjbrandon/models/dnn_model.pth"
    scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
    model_type = "dnn"
    verbose = True
    model, scaler = load_model(model_type, model_path, scaler_path)

    inference = Inference(model, model_type, scaler, verbose)
    intcomm = IntComm(SERIAL_PORT)
    data = []

    print("start")

    start_time = time.time()
    while True:
        yaw, pitch, roll, accx, accy, accz = intcomm.get_acc_gyr_data()
        inference.append_readings(
            yaw, pitch, roll, accx / 8192, accy / 8192, accz / 8192
        )
        result = inference.infer()
        if result:
            print(f"result: {result}")
            if result == "left" or result == "right":
                pass
            else:
                inference.clear()
                end_time = time.time()
                print("response time:", end_time - start_time)
                start_time = end_time
