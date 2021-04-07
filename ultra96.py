import logging
import os
import random
import sys
import threading
import time
import traceback
from queue import SimpleQueue

import pika
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver

from eval_client import Client
from ML import ML

IP_ADDRESS = os.environ["IP_ADDRESS"]
EVAL_PORT = int(os.environ["EVAL_PORT"])
DANCE_PORT = int(os.environ["DANCE_PORT"])
IS_DASHBOARD = bool(os.environ["IS_DASHBOARD"])

WINDOW_SIZE = 50


# setup logging
file_handler = logging.FileHandler(
    filename=f'logs/ultra96_{time.strftime("%Y%m%d-%H%M%S")}.log'
)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("ultra96")


class Server(LineReceiver):
    delimiter = b"\n"

    def __init__(self, persistent_data):
        self.persistent_data = persistent_data

    def connectionMade(self):
        print("New dancer")
        self.persistent_data.num_dancers += 1
        self.printNumDancers()

    def connectionLost(self, reason):
        print("A dancer disconnected")
        self.persistent_data.num_dancers -= 1
        self.printNumDancers()

    def printNumDancers(self):
        print(
            "There are currently %d connected dancers."
            % self.persistent_data.num_dancers
        )

    def lineReceived(self, line):
        line = line.decode()
        try:
            if line[0] != "#":
                logger.error("Received invalid data", line)
                return
            dancer_id, yaw, pitch, roll, accx, accy, accz, emg = line[1:].split(",")

            # appends data for each dancer to window
            dancer_id = int(dancer_id)
            yaw, pitch, roll, accx, accy, accz = (
                float(yaw),
                float(pitch),
                float(roll),
                float(accx),
                float(accy),
                float(accz),
            )
            # TODO: handle start and left and right
            self.persistent_data.ml.write_data(
                dancer_id, [yaw, pitch, roll, accx, accy, accz]
            )

            self.handleMainLogic(dancer_id)
        except Exception:
            logger.error(line)
            logger.error(traceback.format_exc())

    def handleMainLogic(self, dancer_id):
        pred = self.persistent_data.ml.get_pred()
        if pred is not None:
            dance_move, pos, sync_delay = pred
            mqueue.put((dance_move, pos, sync_delay))


# This class is used to store persistent data across connections
class ServerFactory(Factory):
    def __init__(self,):
        self.num_dancers = 0  # number of connected dancers

        self.ml = ML(
            on_fpga=False,
            dance_scaler_path="./dance_scaler.bin",
            pos_scaler_path="./pos_scaler.bin",
            dance_model_path="./dance_model.pth",
            pos_model_path="./pos_model.pth",
        )

    def buildProtocol(self, addr):
        return Server(self)


def swap_positions(positions, pos):
    if pos == ["S", "S", "S"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["R", "L", "S"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["R", "S", "L"]:
        return [positions[2], positions[1], positions[1]]
    elif pos == ["S", "R", "L"]:
        return [positions[0], positions[2], positions[1]]
    elif pos == ["R", "R", "L"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["R", "L", "L"]:
        return [positions[1], positions[2], positions[0]]
    else:
        return [positions[0], positions[1], positions[2]]


def format_results(positions, dance_move, pos, sync_delay):
    new_positions = swap_positions(positions, pos)
    accuracy = random.randrange(60, 100) / 100  # TODO: fixed if got time
    eval_results = f"{new_positions[0]} {new_positions[1]} {new_positions[2]}|{dance_move}|{sync_delay}|"
    dashboard_results = f"{positions[0]} {positions[1]} {positions[2]}|{dance_move}|{new_positions[0]} {new_positions[1]} {new_positions[2]}|{sync_delay}|{accuracy}"

    return eval_results, dashboard_results


if __name__ == "__main__":
    logger.info("Started server on port %d" % DANCE_PORT)

    # setup dashboard queue
    if IS_DASHBOARD:
        CLOUDAMQP_URL = "amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu"
        params = pika.URLParameters(CLOUDAMQP_URL)
        params.socket_timeout = 5
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue="results")

    mqueue = SimpleQueue()
    positions = [1, 2, 3]
    try:
        reactor.listenTCP(DANCE_PORT, ServerFactory())
        thread = threading.Thread(target=reactor.run, args=(False,))
        thread.start()

        input("Press any input to start evaluation server")

        group_id = "18"
        secret_key = "1234123412341234"
        my_client = Client(IP_ADDRESS, EVAL_PORT, group_id, secret_key)
        my_client.send_message("1 2 3" + "|" + "start" + "|" + "1.5" + "|")
        logger.info(f"received positions: {positions}")
        while True:
            while not mqueue.empty():
                dance_move, pos, sync_delay = mqueue.get()
                logger.info(f"predictions: {(dance_move, pos, sync_delay)}")
                eval_results, dashboard_results = format_results(
                    positions, dance_move, pos, sync_delay
                )
                logger.info(f"eval_results: {eval_results}")
                logger.info(f"dashboard_results: {dashboard_results}")

                my_client.send_message(eval_results)
                positions = my_client.receive_dancer_position()
                if IS_DASHBOARD:
                    channel.basic_publish(
                        exchange="", routing_key="results", body=dashboard_results,
                    )
                positions = [int(position) for position in positions.split(" ")]
                logger.info(f"received positions: {positions}")

    except KeyboardInterrupt:
        thread.join()
        logger.info("Terminating")
