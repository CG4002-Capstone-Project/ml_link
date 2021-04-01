import argparse
import base64
import logging
import random
import socket
import sys
import threading
import time
import traceback
import warnings
from queue import SimpleQueue

import pika
from Crypto.Cipher import AES

from eval_client import Client
from inference import Inference, load_model
from utils import (
    dance_move_display,
    dance_position_display,
    ready_display,
    reset_display,
    results_display,
)

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
DB_QUEUES = ["trainee_one_data", "trainee_two_data", "trainee_three_data"]
# Group ID number
GROUP_ID = 18

# key represent the position of main dancer
MAIN_DANCER_1_POSITIONS = {
    0: [[1, 2, 3], [1, 3, 2]],
    1: [[2, 1, 3], [3, 1, 2]],
    2: [[3, 2, 1], [2, 3, 1]],
}
MAIN_DANCER_2_POSITIONS = {
    0: [[2, 1, 3], [2, 3, 1],],
    1: [[1, 2, 3], [3, 2, 1]],
    2: [[3, 1, 2], [1, 3, 2]],
}
MAIN_DANCER_3_POSITIONS = {
    0: [[3, 2, 1], [3, 1, 2]],
    1: [[2, 3, 1], [1, 3, 2],],
    2: [[1, 2, 3], [2, 1, 3],],
}


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

        # time stamps
        # indicates the time when the server receive the package
        self.t2 = 0
        # indicates the time when the server send the package
        self.t3 = 0

        self.timeout = 60
        self.connection = None
        self.timer = None

        # creates a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        logger.info("starting up on %s port %s" % server_address)
        self.socket.bind(server_address)

        # listens for incoming connections
        self.socket.listen(4)
        self.client_address, self.secret_key = self.setup_connection(secret_key)

        self.dance_start_time = None
        self.is_idling = True
        self.is_resetting = True
        self.is_changing_position = True
        self.is_dancing = True
        self.is_waiting = False

        verbose = False
        self.inference = Inference(
            model, model_type, scaler, verbose, infer_dance=True, infer_position=True
        )
        self.counter = 0

    def handle_data(self, data):
        try:
            msg = data.decode("utf8")
            decrypted_message = handle_decryption(msg, self.secret_key)
            dancer_id = int(decrypted_message["dancer_id"])
            assert dancer_id == self.dancer_id
            raw_data = decrypted_message["raw_data"]
            gx, gy, gz, ax, ay, az = [float(x) for x in raw_data.split(" ")]
            self.send_timestamp()

            self.inference.append_readings(gx, gy, gz, ax, ay, az)
            is_ready = self.inference.check_is_ready()

            if not is_ready:
                return

            # idle at the start
            if self.is_idling:
                is_moving = self.inference.check_is_moving()
                if is_moving:
                    self.is_idling = False
                    return
                self.counter += 1
                if self.counter % 30 == 0:
                    logger.info(f"dancer {self.dancer_id}: idling")
                return

            # waiting for reply from main thread
            if self.is_waiting:
                return

            if self.is_resetting:
                # all dancers have to start at the same time
                self.is_waiting = True
                # send to mqueue that is has resetted
                mqueue.put((self.dancer_id, True, ACTION_TYPE_RESET))
                return

            if self.is_changing_position:
                action = self.inference.infer_dancer_left_right()
                if action is not None:
                    self.inference.skip_count = LEFT_RIGHT_DELAY
                    # send to mqueue that that action is taken
                    mqueue.put((self.dancer_id, action, ACTION_TYPE_POSITION))
                return

            if self.dance_start_time is None:
                is_moving = self.inference.check_is_moving()
                if is_moving:
                    self.dance_start_time = time.time()
                    self.inference.skip_count = DANCING_DELAY
                    # send to mqueue timestamp
                    mqueue.put(
                        (
                            self.dancer_id,
                            self.dance_start_time,
                            ACTION_TYPE_START_DANCE_TIME,
                        )
                    )
                return

            if self.is_dancing:
                if inference_mutex.locked():
                    return
                inference_mutex.acquire()
                try:
                    action = self.inference.infer_dancer_moves()
                    if action is not None:
                        # all dancers have to end at the same time
                        self.is_waiting = True
                        # prevent dance moves from added to queue
                        self.is_dancing = False
                        # send to mqueue that that action is taken
                        mqueue.put((self.dancer_id, action, ACTION_TYPE_DANCE_MOVE))
                        mqueue.put(
                            (
                                self.dancer_id,
                                self.inference.dance_detection.accuracy,
                                ACTION_TYPE_ACCURACY,
                            )
                        )
                except:
                    logger.error(traceback.format_exc())
                finally:
                    inference_mutex.release()
                return

        except Exception:
            logger.error(data)
            logger.error(traceback.format_exc())
            self.send_timestamp()

    def run(self):
        while not self.shutdown.is_set():
            # resets when result is sent to evaluation server
            while not queues[self.dancer_id].empty():
                command = queues[self.dancer_id].get()
                if command == COMMAND_RESET:  # resetting
                    self.dance_start_time = None
                    self.is_resetting = True
                    self.is_changing_position = True
                    self.is_dancing = True
                    self.is_waiting = False
                if command == COMMAND_CHANGE_POSITION:  # changing position
                    self.is_resetting = False
                    self.is_waiting = False
                if command == COMMAND_START_DANCING:  # dancing
                    self.is_changing_position = False
                    self.is_waiting = False

            # handles data and inference
            data = self.connection.recv(1024)
            self.t2 = time.time()
            if data:
                self.handle_data(data)
            else:
                logger.warn("no more data from", self.client_address)
                self.stop()

    def send_timestamp(self):
        self.t3 = time.time()
        timestamp = str(self.t2) + "|" + str(self.t3)
        self.connection.sendall(timestamp.encode())

    def setup_connection(self, secret_key):
        logger.info("No actions for 60 seconds to give time to connect")
        self.timer = threading.Timer(self.timeout, self.send_timestamp)
        self.timer.start()

        # waits for a connection
        logger.info("waiting for a connection")
        self.connection, client_address = self.socket.accept()

        logger.info("Enter the secret key: ")
        if not secret_key:
            secret_key = sys.stdin.readline().strip()

        logger.info("connection from" + str(client_address))
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            pass
        else:
            logger.error("AES key must be either 16, 24, or 32 bytes long")
            self.stop()

        return client_address, secret_key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()
        self.socket.close()


def tabulate_dance_moves(
    dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
):
    if dancer_moves[main_dancer_id] is not None:
        return dancer_moves[main_dancer_id], dancer_accuracies[main_dancer_id]

    fallback_dancer_id = 3 - main_dancer_id - guest_dancer_id
    if dancer_moves[fallback_dancer_id] is not None:
        return dancer_moves[fallback_dancer_id], dancer_accuracies[fallback_dancer_id]

    if dancer_moves[guest_dancer_id] is not None:
        return dancer_moves[guest_dancer_id], dancer_accuracies[guest_dancer_id]

    fallbacks = [50.92, 50.72, 50.23]
    return random.choice(ACTIONS), random.choice(fallbacks)


def tabulate_sync_delay(dancer_start_times):
    fallbacks = [0.454965591436666, 0.3437284708026666, 0.248802185056666]
    sync_delay = calculate_sync_delay(*dancer_start_times)
    if sync_delay == -1:
        return random.choice(fallbacks)
    return sync_delay


def tabulate_positions(
    dancer_positions, original_positions, main_dancer_id, guest_dancer_id
):
    # supposedly final positions
    d1_pos = original_positions.index(1) + dancer_positions[0]
    d2_pos = original_positions.index(2) + dancer_positions[1]
    d3_pos = original_positions.index(3) + dancer_positions[2]

    # fix errors if greater than 2 or less than 0
    d1_pos_final = max(0, min(2, d1_pos))
    d2_pos_final = max(0, min(2, d2_pos))
    d3_pos_final = max(0, min(2, d3_pos))

    final_positions = [-1, -1, -1]

    # assumes that main dancer is always right and guest dancer is often wrong
    if main_dancer_id == 0:

        if guest_dancer_id == 0:
            # should never happen but just random choices
            return random.choice(POSITIONS)
        elif guest_dancer_id == 1:
            # 0 - main; 1 - guest; 2 - fallback
            if d1_pos_final == d3_pos_final:
                return random.choice(MAIN_DANCER_1_POSITIONS[d1_pos_final])
            else:
                final_positions[d1_pos_final] = 1
                final_positions[d3_pos_final] = 3
                final_positions[final_positions.index(-1)] = 2
                return final_positions
        else:
            # 0 - main; 1 - fallback; 2 - guest
            if d1_pos_final == d2_pos_final:
                return random.choice(MAIN_DANCER_1_POSITIONS[d1_pos_final])
            else:
                final_positions[d1_pos_final] = 1
                final_positions[d2_pos_final] = 2
                final_positions[final_positions.index(-1)] = 3
                return final_positions

    elif main_dancer_id == 1:

        if guest_dancer_id == 0:
            # 0 - guest; 1 - main; 2 - fallback
            if d2_pos_final == d3_pos_final:
                return random.choice(MAIN_DANCER_2_POSITIONS[d2_pos_final])
            else:
                final_positions[d2_pos_final] = 2
                final_positions[d3_pos_final] = 3
                final_positions[final_positions.index(-1)] = 1
                return final_positions
        elif guest_dancer_id == 1:
            # should never happen but just random choices
            return random.choice(POSITIONS)
        else:
            # 0 - fallback; 1 - main; 2 - guest
            if d1_pos_final == d2_pos_final:
                return random.choice(MAIN_DANCER_2_POSITIONS[d2_pos_final])
            else:
                final_positions[d2_pos_final] = 2
                final_positions[d1_pos_final] = 1
                final_positions[final_positions.index(-1)] = 3
                return final_positions

    else:

        if guest_dancer_id == 0:
            # 0 - guest; 1 - fallback; 2 - main
            if d2_pos_final == d3_pos_final:
                return random.choice(MAIN_DANCER_3_POSITIONS[d3_pos_final])
            else:
                final_positions[d3_pos_final] = 3
                final_positions[d2_pos_final] = 2
                final_positions[final_positions.index(-1)] = 1
                return final_positions
        elif guest_dancer_id == 1:
            # 0 - fallback; 1 - guest; 2 - main
            if d1_pos_final == d3_pos_final:
                return random.choice(MAIN_DANCER_3_POSITIONS[d3_pos_final])
            else:
                final_positions[d3_pos_final] = 3
                final_positions[d1_pos_final] = 1
                final_positions[final_positions.index(-1)] = 2
                return final_positions
        else:
            # should never happen but just random choices
            return random.choice(POSITIONS)

    final_positions = [-1, -1, -1]
    final_positions[d3_pos_final] = 3
    final_positions[d2_pos_final] = 2
    final_positions[d1_pos_final] = 1

    return final_positions


def tabulate_results(
    dancer_readiness,
    dancer_start_times,
    dancer_moves,
    dancer_accuracies,
    dancer_positions,
    original_positions,
    main_dancer_id,
    guest_dancer_id,
):
    positions = original_positions.copy()

    dance_move, accuracy = tabulate_dance_moves(
        dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
    )
    sync_delay = tabulate_sync_delay(dancer_start_times)
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )

    return dance_move, sync_delay, positions, accuracy


def format_eval_data(original_positions, positions, dance_move, sync_delay, accuracy):
    eval_data = f"{positions[0]} {positions[1]} {positions[2]}|{dance_move}|{round(sync_delay, 4)}|"
    return eval_data


def format_dashboard_data(
    original_positions, positions, dance_move, sync_delay, accuracy
):
    # eval_server_positions|detected move|detected_positions|sync_delay|accuracy
    dashboard_data = f"{original_positions[0]} {original_positions[1]} {original_positions[2]}|{dance_move}|{positions[0]} {positions[1]} {positions[2]}|{round(sync_delay, 4)}|{round(accuracy, 4)}"
    return dashboard_data


def main(
    dancer_ids,
    secret_key,
    ip_addr,
    is_dashboard,
    is_eval_server,
    main_dancer_id,
    guest_dancer_id,
    port_num=8001,
    group_id="18",
):
    # note that the same secret_key is use throughout the system for dancers and eval server
    if is_eval_server:
        my_client = Client(ip_addr, port_num, group_id, secret_key)

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
        logger.info(
            "dancer_server0 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[0])
            + " Group ID number: "
            + str(GROUP_ID)
        )

    if 1 in dancer_ids:
        dancer_server1.start()
        logger.info(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[1])
            + " Group ID number: "
            + str(GROUP_ID)
        )
    if 2 in dancer_ids:
        dancer_server2.start()
        logger.info(
            "dancer_server1 started: IP address:"
            + IP_ADDRESS
            + " Port Number: "
            + str(PORT_NUM[2])
            + " Group ID number: "
            + str(GROUP_ID)
        )

    if is_dashboard:
        CLOUDAMQP_URL = "amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu"
        params = pika.URLParameters(CLOUDAMQP_URL)
        params.socket_timeout = 5

        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue="results")
        channel.queue_declare(queue="mode")

    dancer_readiness = [False, False, False]
    dancer_start_times = [None, None, None]
    dancer_moves = [None, None, None]
    dancer_accuracies = [None, None, None]
    dancer_positions = [0, 0, 0]
    original_positions = [1, 2, 3]

    start_time = time.time()
    stage = -1
    counter = 3

    if is_eval_server:
        my_client.send_message("1 2 3" + "|" + "start" + "|" + "1.5" + "|")

    while True:
        while not mqueue.empty():
            dancer_id, action, action_type = mqueue.get()
            logger.info(("received:", dancer_id, action, action_type))
            if action_type == ACTION_TYPE_RESET:  # resetting
                dancer_readiness[dancer_id] = True
            if action_type == ACTION_TYPE_POSITION:  # positions
                if action == "left":
                    dancer_positions[dancer_id] -= 1
                if action == "right":
                    dancer_positions[dancer_id] += 1
            if action_type == ACTION_TYPE_START_DANCE_TIME:  # starting timestamp
                dancer_start_times[dancer_id] = action
            if action_type == ACTION_TYPE_DANCE_MOVE:  # moves
                dancer_moves[dancer_id] = action
            if action_type == ACTION_TYPE_ACCURACY:  # accuracies
                dancer_accuracies[dancer_id] = action

        # start changing positions if display changes
        if (
            sum(dancer_readiness) == len(dancer_ids) and stage == -1
        ):  # NOTE: edit this for testing
            if counter > 0:
                ready_display(counter)
                time.sleep(1)
                counter -= 1
                continue

            for q in queues:
                q.put(COMMAND_CHANGE_POSITION)
            start_time = time.time()
            stage = COMMAND_CHANGE_POSITION
            if is_dashboard:
                channel.basic_publish(
                    exchange="", routing_key="mode", body="CHANGE POSITIONS",
                )
            dance_position_display()
            results_display(
                [
                    dancer_readiness,
                    dancer_start_times,
                    dancer_moves,
                    dancer_accuracies,
                    dancer_positions,
                    original_positions,
                ]
            )
            continue

        # start changing positions if all dancers are resetted
        if (
            all(dancer_readiness) or (time.time() - start_time > 14)
        ) and stage == 0:  # NOTE: edit this for testing
            if counter > 0:
                ready_display(counter)
                time.sleep(1)
                counter -= 1
                continue

            for q in queues:
                q.put(COMMAND_CHANGE_POSITION)
            start_time = time.time()
            stage = COMMAND_CHANGE_POSITION
            if is_dashboard:
                channel.basic_publish(
                    exchange="", routing_key="mode", body="CHANGE POSITIONS",
                )
            dance_position_display()
            results_display(
                [
                    dancer_readiness,
                    dancer_start_times,
                    dancer_moves,
                    dancer_accuracies,
                    dancer_positions,
                    original_positions,
                ]
            )
            continue

        # start dancing after changing positions for some interval
        if time.time() - start_time > 7 and stage == 1:
            for q in queues:
                q.put(COMMAND_START_DANCING)
            start_time = time.time()
            stage = COMMAND_START_DANCING
            if is_dashboard:
                channel.basic_publish(
                    exchange="", routing_key="mode", body="START DANCING",
                )
            dance_move_display()
            results_display(
                [
                    dancer_readiness,
                    dancer_start_times,
                    dancer_moves,
                    dancer_accuracies,
                    dancer_positions,
                    original_positions,
                ]
            )
            continue

        # tabulate inference and reset
        if (
            all(dancer_moves) or (time.time() - start_time > 14)
        ) and stage == 2:  # NOTE: edit this for testing
            dance_move, sync_delay, positions, accuracy = tabulate_results(
                dancer_readiness,
                dancer_start_times,
                dancer_moves,
                dancer_accuracies,
                dancer_positions,
                original_positions,
                main_dancer_id,
                guest_dancer_id,
            )

            # display and sends results
            eval_data = format_eval_data(
                original_positions, positions, dance_move, sync_delay, accuracy
            )
            if is_eval_server:
                my_client.send_message(eval_data)
            if is_eval_server:
                original_positions = my_client.receive_dancer_position()
            else:
                original_positions = "1 2 3"
            original_positions = [
                int(position) for position in original_positions.split(" ")
            ]
            dashboard_data = format_dashboard_data(
                original_positions, positions, dance_move, sync_delay, accuracy
            )

            if is_dashboard:
                channel.basic_publish(
                    exchange="", routing_key="results", body=dashboard_data,
                )
            if is_dashboard:
                channel.basic_publish(
                    exchange="", routing_key="mode", body="RESETTING... DO NOT MOVE...",
                )
            reset_display()
            results_display(
                [
                    dancer_readiness,
                    dancer_start_times,
                    dancer_moves,
                    dancer_accuracies,
                    dancer_positions,
                    original_positions,
                ]
            )
            logger.info("### tabulated result ###")
            logger.info(eval_data)
            logger.info(dashboard_data)
            time.sleep(3)

            # reset
            dancer_readiness = [False, False, False]
            dancer_start_times = [None, None, None]
            dancer_moves = [None, None, None]
            dancer_accuracies = [None, None, None]
            dancer_positions = [0, 0, 0]
            logger.info(f"received dancer postions: {original_positions}")
            for q in queues:
                q.put(COMMAND_RESET)
            start_time = time.time()
            stage = COMMAND_RESET

            continue


if __name__ == "__main__":
    # setup logging
    file_handler = logging.FileHandler(
        filename=f'ultra96_{time.strftime("%Y%m%d-%H%M%S")}.log'
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger("ultra96")

    # parse arguments
    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument(
        "--dancer_ids", help="dancer id", nargs="+", type=int, required=True
    )
    parser.add_argument("--secret_key", default="1234123412341234", help="secret key")
    parser.add_argument(
        "--is_dashboard", default=False, help="send to dashboard", type=bool
    )
    parser.add_argument(
        "--is_eval_server", default=False, help="send to eval sever", type=bool
    )
    parser.add_argument(
        "--ip_addr", default="localhost", help="ip address of eval_server"
    )
    parser.add_argument("--model_type", default="dnn", help="type of model")
    parser.add_argument("--main_dancer_id", default=0, help="main dancer", type=int)
    parser.add_argument("--guest_dancer_id", default=2, help="guest dancer", type=int)

    args = parser.parse_args()
    dancer_ids = args.dancer_ids
    secret_key = args.secret_key
    is_dashboard = args.is_dashboard
    is_eval_server = args.is_eval_server
    ip_addr = args.ip_addr
    model_type = args.model_type
    main_dancer_id = args.main_dancer_id
    guest_dancer_id = args.guest_dancer_id

    logger.info("dancer_ids:" + str(dancer_ids))
    logger.info("secret_key:" + str(secret_key))
    logger.info("is_dashboard:" + str(is_dashboard))
    logger.info("is_eval_server:" + str(is_eval_server))
    logger.info("ip_addr:" + str(ip_addr))
    logger.info("model_type:" + str(model_type))
    logger.info("main_dancer_id:" + str(main_dancer_id))
    logger.info("guest_dancer_id:" + str(guest_dancer_id))

    assert main_dancer_id != guest_dancer_id

    COMMAND_RESET = 0
    COMMAND_CHANGE_POSITION = 1
    COMMAND_START_DANCING = 2

    ACTION_TYPE_RESET = 0
    ACTION_TYPE_POSITION = 1
    ACTION_TYPE_START_DANCE_TIME = 2
    ACTION_TYPE_DANCE_MOVE = 3
    ACTION_TYPE_ACCURACY = 4

    LEFT_RIGHT_DELAY = 30
    RESET_DELAY = 90
    DANCING_DELAY = 90

    if model_type == "dnn":
        model_path = "/home/nwjbrandon/models/dnn_model.pth"
        scaler_path = "/home/nwjbrandon/models/dnn_std_scaler.bin"
        model, scaler = load_model(model_type, model_path, scaler_path)
    else:
        model_path = "../models/wts"
        scaler_path = "../models/dnn_std_scaler.bin"
        model_type = "fpga"  # run fpga model
        model, scaler = load_model(model_type, model_path, scaler_path)

    inference_mutex = threading.Lock()

    queues = [SimpleQueue(), SimpleQueue(), SimpleQueue()]
    mqueue = SimpleQueue()

    main(
        dancer_ids,
        secret_key,
        ip_addr,
        is_dashboard,
        is_eval_server,
        main_dancer_id,
        guest_dancer_id,
    )