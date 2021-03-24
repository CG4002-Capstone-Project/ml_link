import time

from inference import Inference, load_model
from intcomm import SERIAL_PORT, IntComm

if __name__ == "__main__":
    model_path = "/home/nwjbrandon/models/dnn_model.pth"
    scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
    model_type = "dnn"
    verbose = True
    model, scaler = load_model(model_type, model_path, scaler_path)

    inference = Inference(model, model_type, scaler, verbose, infer_dance=True)
    intcomm = IntComm(SERIAL_PORT)
    data = []

    start_time = time.time()
    while True:
        gx, gy, gz, ax, ay, az = intcomm.get_acc_gyr_data()
        inference.append_readings(gx, gy, gz, ax, ay, az)
        result = inference.infer()
        if result:
            print(f"result: {result}")
            if not (result == "left" or result == "right"):
                inference.clear()
                end_time = time.time()
                print("response time:", int(end_time - start_time))
                start_time = end_time
