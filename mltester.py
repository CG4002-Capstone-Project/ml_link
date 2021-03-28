import time

from inference import Inference, load_model
from intcomm import IntComm

if __name__ == "__main__":
    model_path = "/home/nwjbrandon/models/dnn_model.pth"
    scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
    model_type = "dnn"
    verbose = True
    model, scaler = load_model(model_type, model_path, scaler_path)

    inference = Inference(
        model, model_type, scaler, verbose, infer_dance=False, infer_position=True
    )
    # change this according to your serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0)
    data = []

    start_time = time.time()
    while True:
        gx, gy, gz, ax, ay, az = intcomm.get_acc_gyr_data()
        inference.append_readings(gx, gy, gz, ax, ay, az)

        is_ready = inference.check_is_ready()
        if not is_ready:
            continue

        # # left or right
        # action = inference.infer_dancer_left_right()
        # if action is not None:
        #     print(action)
        #     inference.skip_count = 30

        # dance move
        action = inference.infer_dancer_moves()
        if action is not None:
            print(action)
            inference.skip_count = 60
