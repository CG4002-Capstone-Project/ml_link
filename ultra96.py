import argparse
import base64
import socket
import sys
import threading
import time
import traceback
import warnings
from queue import SimpleQueue

from Crypto.Cipher import AES

from inference import Inference, load_model

warnings.filterwarnings("ignore")
# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
# Week 9 and 11 tests: 3 moves, repeated 4 times each = 12 moves.
ACTIONS = ["gun", "sidepump", "hair"]
POSITIONS = ["1 2 3", "3 2 1", "2 3 1", "3 1 2", "1 3 2", "2 1 3"]
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 4  # dancer_id, t1, offset and raw_data
ENCRYPT_BLOCK_SIZE = 16

# The IP address of the Ultra96, testing part will be "127.0.0.1"
IP_ADDRESS = "127.0.0.1"
# The port number for three different dancer's laptops
PORT_NUM = [9091, 9092, 9093]
# Group ID number
GROUP_ID = 18


def calculate_ultra96_time(beetle_time, offset):
    return beetle_time - offset


def calculate_sync_delay(start_time0, start_time1, start_time2):
    if start_time0 and start_time1 and start_time2:
        return abs(
            max(start_time0, start_time1, start_time2)
            - min(start_time0, start_time1, start_time2)
        )
    elif start_time0 and start_time1:
        return abs(max(start_time0, start_time1) - min(start_time0, start_time1))
    elif start_time0 and start_time2:
        return abs(max(start_time0, start_time2) - min(start_time0, start_time2))
    elif start_time1 and start_time2:
        return abs(max(start_time1, start_time2) - min(start_time1, start_time2))
    else:
        return -1


def handle_decryption(cipher_text, secret_key):
    # data format: raw data | t0 | RTT | offset | start_flag | muscle_fatigue
    decoded_message = base64.b64decode(cipher_text)
    iv = decoded_message[:16]
    secret_key = bytes(str(secret_key), encoding="utf8")

    cipher = AES.new(secret_key, AES.MODE_CBC, iv)
    decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
    decrypted_message = decrypted_message.decode("utf8")

    decrypted_message = decrypted_message[decrypted_message.find("#") :]
    decrypted_message = bytes(decrypted_message[1:], "utf8").decode("utf8")

    messages = decrypted_message.split("|")

    dancer_id, t1, offset, raw_data = messages[:MESSAGE_SIZE]
    return {
        "dancer_id": dancer_id,
        "t1": t1,
        "offset": offset,
        "raw_data": raw_data,
    }


class Server(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, secret_key, dancer_id):
        super(Server, self).__init__()
        self.dancer_id = dancer_id

        # Time stamps
        # Indicate the time when the server receive the package
        self.t2 = 0
        # Indicate the time when the server send the package
        self.t3 = 0

        self.timeout = 60
        self.connection = None
        self.timer = None

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print("starting up on %s port %s" % server_address)
        self.socket.bind(server_address)

        # Listen for incoming connections
        self.socket.listen(4)
        self.client_address, self.secret_key = self.setup_connection(secret_key)

        self.dance_start_time = None

        model_path = "/home/nwjbrandon/models/dnn_model.pth"
        scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
        model_type = "dnn"
        model, scaler = load_model(model_type, model_path, scaler_path)
        verbose = False
        self.inference = Inference(
            model, model_type, scaler, verbose, infer_dance=True, infer_position=True
        )

    def handle_data(self, data):
        try:
            msg = data.decode("utf8")
            decrypted_message = handle_decryption(msg, self.secret_key)
            dancer_id = int(decrypted_message["dancer_id"])
            assert dancer_id == self.dancer_id

            # send start dance time once
            if (
                (not self.inference.is_still)
                and (not self.dance_start_time)
                and (not self.inference.is_idling)
            ):
                self.dance_start_time = calculate_ultra96_time(
                    float(decrypted_message["t1"]), float(decrypted_message["offset"]),
                )
                mqueue.put(
                    (self.dancer_id, self.dance_start_time, 0)
                )  # 0 indicates sync delay message

            raw_data = decrypted_message["raw_data"]
            gx, gy, gz, ax, ay, az = [float(x) for x in raw_data.split(" ")]

            self.inference.append_readings(gx, gy, gz, ax, ay, az)
            result = self.inference.infer()
            if result is not None:
                mqueue.put(
                    (self.dancer_id, result, 1)
                )  # 1 indicates dance moves or positions
            self.send_timestamp()

        except Exception:
            print(data)
            print(traceback.format_exc())
            self.send_timestamp()

    def run(self):
        while not self.shutdown.is_set():
            # resets when result is sent to evaluation server
            while not queues[self.dancer_id].empty():
                queues[self.dancer_id].get()
                print("reseting", self.dancer_id)
                self.inference.is_still = True
                self.dance_start_time = None

            # handles data and inference
            data = self.connection.recv(1024)
            self.t2 = time.time()
            if data:
                self.handle_data(data)
            else:
                print("no more data from", self.client_address)
                self.stop()

    def send_timestamp(self):
        self.t3 = time.time()
        timestamp = str(self.t2) + "|" + str(self.t3)
        self.connection.sendall(timestamp.encode())

    def setup_connection(self, secret_key):
        print("No actions for 60 seconds to give time to connect")
        self.timer = threading.Timer(self.timeout, self.send_timestamp)
        self.timer.start()

        # waits for a connection
        print("waiting for a connection")
        self.connection, client_address = self.socket.accept()

        print("Enter the secret key: ")
        if not secret_key:
            secret_key = sys.stdin.readline().strip()

        print("connection from", client_address)
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            pass
        else:
            print("AES key must be either 16, 24, or 32 bytes long")
            self.stop()

        return client_address, secret_key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()
        self.socket.close()


def tabulate_results(
    original_positions,
    dance_start_times,
    dance_moves,
    dance_positions,
    main_dancer_id=0,
    guest_dancer_id=2,
):
    positions = original_positions.copy()  # make a copy of the original positions

    dance_move = dance_moves[main_dancer_id]
    sync_delay = calculate_sync_delay(*dance_start_times)
    positions[0] += dance_positions[0]
    positions[1] += dance_positions[1]
    positions[2] += dance_positions[2]

    return dance_move, sync_delay, positions


def main(dancer_ids, secret_key):
    if 0 in dancer_ids:
        dancer_server0 = Server(
            IP_ADDRESS, PORT_NUM[0], GROUP_ID, secret_key, dancer_id=0
        )
    if 1 in dancer_ids:
        dancer_server1 = Server(
            IP_ADDRESS, PORT_NUM[1], GROUP_ID, secret_key, dancer_id=1
        )
    if 2 in dancer_ids:
        dancer_server2 = Server(
            IP_ADDRESS, PORT_NUM[2], GROUP_ID, secret_key, dancer_id=2
        )

    if 0 in dancer_ids:
        dancer_server0.start()
        print(
            "dancer_server0 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[0])
            + " Group ID number: "
            + str(GROUP_ID)
        )

    if 1 in dancer_ids:
        dancer_server1.start()
        print(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[1])
            + " Group ID number: "
            + str(GROUP_ID)
        )
    if 2 in dancer_ids:
        dancer_server2.start()
        print(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[2])
            + " Group ID number: "
            + str(GROUP_ID)
        )

    counter = 0
    dance_start_times = [None, None, None]  # timestamp of start of dancer dacing
    dance_moves = [None, None, None]  # dance moves of dancer
    dance_positions = [0, 0, 0]  # displacements of dancer
    original_positions = [1, 2, 3]
    while True:
        time.sleep(0.5)

        while not mqueue.empty():
            dancer_id, action, action_type = mqueue.get()
            print("received:", dancer_id, action, action_type)
            # sets dance start time
            if action_type == 0:
                dance_start_times[dancer_id] = action
            # sets dance position and moves
            if action_type == 1:
                if action == "left":
                    dance_positions[dancer_id] -= 1
                if action == "right":
                    dance_positions[dancer_id] += 1
                if action in ACTIONS:
                    dance_moves[dancer_id] = action

        # tabulate results when all is filled
        if all(dance_start_times[:2]) and all(dance_moves[:2]):
            dance_move, sync_delay, positions = tabulate_results(
                original_positions,
                dance_start_times,
                dance_moves,
                dance_positions,
                main_dancer_id=0,
                guest_dancer_id=2,
            )
            print("sending:", dance_move, sync_delay, positions)
            # reset
            counter = 0
            dance_start_times = [None, None, None]  # timestamp of start of dancer dacing
            dance_moves = [None, None, None]  # dance moves of dancer
            dance_positions = [0, 0, 0]  # displacements of dancer
            for q in queues:
                q.put(1)

        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument(
        "--dancer_ids", help="dancer id", nargs="+", type=int, required=True
    )
    parser.add_argument("--secret_key", default="1234123412341234", help="secret key")

    args = parser.parse_args()
    dancer_ids = args.dancer_ids
    secret_key = args.secret_key

    print("dancer_id:", dancer_ids)
    print("secret_key:", secret_key)

    queues = [SimpleQueue(), SimpleQueue(), SimpleQueue()]
    mqueue = SimpleQueue()

    main(dancer_ids, secret_key)
