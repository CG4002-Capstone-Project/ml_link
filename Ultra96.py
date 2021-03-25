import argparse
import base64
import random
import socket
import sys
import threading
import time
import traceback
import warnings

import numpy as np
import pika
from Crypto import Random
from Crypto.Cipher import AES
from joblib import load
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

class Dancers:

    def __init__(self):
        model_path = "/home/nwjbrandon/models/dnn_model.pth"
        scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
        model_type = "dnn"
        verbose = False
        model, scaler = load_model(model_type, model_path, scaler_path)

        self.inference0 = Inference(model, model_type, scaler, verbose, infer_dance=True, infer_position=True)
        self.inference1 = Inference(model, model_type, scaler, verbose, infer_dance=True, infer_position=True)
        self.inference2 = Inference(model, model_type, scaler, verbose, infer_dance=True, infer_position=True)

    def append_readings(self, dancer_id, gx, gy, gz, ax, ay, az):
        if dancer_id == 0:
            self.inference0.append_readings(gx, gy, gz, ax, ay, az)
        elif dancer_id == 1:
            self.inference1.append_readings(gx, gy, gz, ax, ay, az)
        elif dancer_id == 2:
            self.inference2.append_readings(gx, gy, gz, ax, ay, az)
        else:
            raise Exception("dancer_id not defined")

    def get_is_still(self, dancer_id):
        if dancer_id == 0:
            return self.inference0.is_still
        elif dancer_id == 1:
            return self.inference1.is_still
        elif dancer_id == 2:
            return self.inference2.is_still
        else:
            raise Exception("dancer_id not defined")

    def get_dance_time(self, dancer_id):
        if dancer_id == 0:
            return self.inference0.dance_time
        elif dancer_id == 1:
            return self.inference1.dance_time
        elif dancer_id == 2:
            return self.inference2.dance_time
        else:
            raise Exception("dancer_id not defined")

    def get_inference(self, dancer_id):
        if dancer_id == 0:
            return self.inference0.infer()
        elif dancer_id == 1:
            return self.inference1.infer()
        elif dancer_id == 2:
            return self.inference2.infer()
        else:
            raise Exception("dancer_id not defined")

    def set_skip_count(self, dancer_id, skip_count):
        if dancer_id == 0:
            self.inference0.skip_count = skip_count
        elif dancer_id == 1:
            self.inference1.skip_count = skip_count
        elif dancer_id == 2:
            self.inference2.skip_count = skip_count
        else:
            raise Exception("dancer_id not defined")

    def set_dance_time(self, dancer_id, dance_time):
        if dancer_id == 0:
            self.inference0.dance_time = dance_time
        elif dancer_id == 1:
            self.inference1.dance_time = dance_time
        elif dancer_id == 2:
            self.inference2.dance_time = dance_time
        else:
            raise Exception("dancer_id not defined")      



class Server(threading.Thread):
    def __init__(
        self,
        ip_addr,
        port_num,
        group_id,
        secret_key,
        dancers,
        n_moves=len(ACTIONS) * NUM_MOVE_PER_ACTION,
    ):
        super(Server, self).__init__()

        # Time stamps
        # Indicate the time when the server receive the package
        self.t2 = 0
        # Indicate the time when the server send the package
        self.t3 = 0

        # setup moves
        self.actions = ACTIONS
        self.position = POSITIONS
        self.n_moves = int(n_moves)

        # the moves should be a random distribution
        self.move_idxs = list(range(self.n_moves)) * NUM_MOVE_PER_ACTION
        assert self.n_moves == len(self.actions) * NUM_MOVE_PER_ACTION
        self.action = None
        self.action_set_time = None

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.dancer_positions = ["1", "2", "3"]
        self.sync_delay = 0

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print("starting up on %s port %s" % server_address)
        self.socket.bind(server_address)

        # Listen for incoming connections
        self.socket.listen(4)
        self.client_address, self.secret_key = self.setup_connection(secret_key)

        self.BUFFER = []
        self.idle_state = True
        self.idle_index = 0
        self.dancers = dancers



    def decrypt_message(self, cipher_text):
        # The data format which will be used here will be "raw data | t0 | RTT | offset | start_flag | muscle_fatigue"
        decoded_message = base64.b64decode(cipher_text)
        iv = decoded_message[:16]
        secret_key = bytes(str(self.secret_key), encoding="utf8")

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

    def calculate_Ultra96_time(self, beetle_time, offset):
            return (beetle_time - offset)

    def calculate_sync_delay(self,start_time0, start_time1, start_time2):
        print("calculating sync delay")
        if start_time0 != None and start_time1 != None and start_time2 != None:
            return abs(max(start_time0, start_time1, start_time2) - min(start_time0, start_time1, start_time2))
            
        elif start_time1 != None and start_time2 != None:
            return abs(max(start_time1, start_time2) - min(start_time1, start_time2))
        elif start_time0 != None and start_time2 != None:
            return abs(max(start_time0, start_time2) - min(start_time0, start_time2))
        elif start_time0 != None and start_time1 != None:
            return abs(max(start_time0, start_time1) - min(start_time0, start_time1))
        else:
            #Magic number to test, change to 0 afterwards
            return 3.1415926
            

    def run(self):
        # global inference0, inference1

        while not self.shutdown.is_set():
            data = self.connection.recv(1024)
            self.t2 = time.time()
            if verbose:
                print("t2:" + str(self.t2))

            if data:
                try:
                    msg = data.decode("utf8")
                    decrypted_message = self.decrypt_message(msg)
                    dancer_id = int(decrypted_message["dancer_id"])

                    if self.dancers.get_is_still(dancer_id) == False and self.dancers.get_dance_time(dancer_id) == None:
                        dance_time = self.calculate_Ultra96_time(float(decrypted_message["t1"]), float(decrypted_message["offset"])) 
                        self.dancers.set_dance_time(dancer_id, dance_time)


                    if self.dancers.get_dance_time(0) != None and self.dancers.get_dance_time(1) != None:
                        sync_delay = self.calculate_sync_delay(self.dancers.get_dance_time(0), self.dancers.get_dance_time(1), None)
                        self.dancers.set_dance_time(0, None)
                        self.dancers.set_dance_time(1, None)
                        print("sync delay", sync_delay)
                        self.dancers.set_skip_count(0, 60)
                        self.dancers.set_skip_count(1, 60)
                        
                    raw_data = decrypted_message["raw_data"]
                    gx, gy, gz, ax, ay, az  = [float(x) for x in raw_data.split(" ")]

                    self.dancers.append_readings(dancer_id, gx, gy, gz, ax, ay, az)
                    result = self.dancers.get_inference(dancer_id)

                    self.send_timestamp()

                except Exception:
                    print(traceback.format_exc())
            else:
                print("no more data from", self.client_address)
                self.stop()


    def send_timestamp(self):
        self.t3 = time.time()
        timestamp = str(self.t2) + "|" + str(self.t3)

        if verbose:
            print("t3:" + str(self.t3))
            print("timestamp: " + timestamp)

        self.connection.sendall(timestamp.encode())

    def setup_connection(self, secret_key):
        random.shuffle(self.move_idxs)
        print("No actions for 60 seconds to give time to connect")
        self.timer = threading.Timer(self.timeout, self.send_timestamp)
        self.timer.start()

        # Wait for a connection
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

        return client_address, secret_key  # forgot to return the secret key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()


def main(dancer_id, secret_key):
    dancers = Dancers()

    dancer_server0 = Server(IP_ADDRESS, PORT_NUM[0], GROUP_ID, secret_key, dancers)
    dancer_server1 = Server(IP_ADDRESS, PORT_NUM[1], GROUP_ID, secret_key, dancers)
    # dancer_server2 = Server(IP_ADDRESS, PORT_NUM[2], GROUP_ID, secret_key)

    dancer_server0.start()
    print(
        "dancer_server0 started: IP address:"
        + IP_ADDRESS
        + " Port Number: "
        + str(PORT_NUM[0])
        + " Group ID number: "
        + str(GROUP_ID)
    )

    dancer_server1.start()
    print(
        "dancer_server1 started: IP address:"
        + IP_ADDRESS
        + " Port Number: "
        + str(PORT_NUM[1])
        + " Group ID number: "
        + str(GROUP_ID)
    )
    # if dancer_id == 2:
    #     dancer_server2.start()
    #     print(
    #         "dancer_server1 started: IP address:"
    #         + IP_ADDRESS
    #         + " Port Number: "
    #         + str(PORT_NUM[2])
    #         + " Group ID number: "
    #         + str(GROUP_ID)
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument("--dancer_id", help="dancer id", type=int, required=True)
    parser.add_argument("--debug", default=False, help="debug mode", type=bool)
    parser.add_argument(
        "--production", default=False, help="production mode", type=bool
    )
    parser.add_argument("--verbose", default=False, help="verbose", type=bool)
    parser.add_argument("--model_type", help="svc or dnn or ultra96 model")
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--scaler_path", help="path to scaler")
    parser.add_argument("--secret_key", default="1234123412341234", help="secret key")
    parser.add_argument(
        "--bit_path",
        default="/home/xilinx/jupyter_notebooks/frontier/capstone_full.bit",
        help="path to bit",
    )
    parser.add_argument(
        "--eval_server", default=False, help="send to eval server", type=bool
    )
    parser.add_argument("--ip_addr", default="localhost", help="eval server ip")
    parser.add_argument("--port_num", default="8000", help="eval server port", type=int)
    parser.add_argument("--group_id", default="18", help="group number")
    parser.add_argument(
        "--key", default="1234123412341234", help="secret key", type=int
    )
    parser.add_argument(
        "--dashboard", default=False, help="send to dashboard", type=bool
    )
    parser.add_argument(
        "--cloudamqp_url",
        default="amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu",
        help="dashboard connection",
    )

    args = parser.parse_args()
    dancer_id = args.dancer_id
    debug = args.debug
    production = args.production
    verbose = args.verbose
    model_type = args.model_type
    model_path = args.model_path
    scaler_path = args.scaler_path
    secret_key = args.secret_key
    bit_path = args.bit_path
    eval_server = args.eval_server
    ip_addr = args.ip_addr
    port_num = args.port_num
    group_id = args.group_id
    key = args.key
    dashboard = args.dashboard
    cloudamqp_url = args.cloudamqp_url

    print("dancer_id:", dancer_id)
    print("debug:", debug)
    print("production:", production)
    print("verbose:", verbose)
    print("model_type:", model_type)
    print("model_path:", model_path)
    print("scaler_path:", scaler_path)
    print("secret_key:", secret_key)
    print("eval_server:", eval_server)
    print("bit_path:", bit_path)
    print("ip_addr:", ip_addr)
    print("port_num:", port_num)
    print("group_id:", group_id)
    print("key:", key)
    print("dashboard:", dashboard)
    print("cloudamqp_url:", cloudamqp_url)

    # model_path = "/home/nwjbrandon/models/dnn_model.pth"
    # scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
    # model_type = "dnn"
    # verbose = False
    # model, scaler = load_model(model_type, model_path, scaler_path)

    # inference0 = Inference(model, model_type, scaler, verbose, infer_dance=True, infer_position=True)
    # inference1 = Inference(model, model_type, scaler, verbose, infer_dance=True, infer_position=True)

    main(dancer_id, secret_key)
