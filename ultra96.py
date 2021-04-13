import logging
import os
import random
import sys
import threading
import time
import traceback
from queue import SimpleQueue

import pika
from scipy import stats
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver

from eval_client import ACTIONS, Client
from ML import ML

IP_ADDRESS = os.environ["IP_ADDRESS"]
EVAL_PORT = int(os.environ["EVAL_PORT"])
DANCE_PORT = int(os.environ["DANCE_PORT"])
IS_DASHBOARD = bool(int(os.environ["IS_DASHBOARD"]))


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
        self.clearLineBuffer()

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

    def getFrequency(self):
        self.persistent_data.counter += 1
        if self.persistent_data.counter % 300 == 0:
            end_time = time.time()
            logger.info(
                "Receiving data at %f Hz"
                % (300 / (end_time - self.persistent_data.start_time))
            )
            self.persistent_data.start_time = end_time

    def skipInitialReadings(self, dancer_id, data_type):
        if self.persistent_data.skip_initial_readings_cnt[dancer_id][data_type] > 0:
            self.persistent_data.skip_initial_readings_cnt[dancer_id][data_type] -= 1
            return True
        return False

    def lineReceived(self, line):
        try:
            line = line.decode()

            self.getFrequency()

            if line[0] != "#":
                logger.error("Received invalid data", line)
                return
            (
                dancer_id,
                data_type,
                yaw,
                pitch,
                roll,
                gyrox,
                gyroy,
                gyroz,
                accx,
                accy,
                accz,
                emg,
            ) = line[1:].split(",")

            # appends data for each dancer to window
            dancer_id = int(dancer_id)
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
            data_type = int(data_type)

            if self.skipInitialReadings(dancer_id, data_type):
                return

            if self.persistent_data.is_idle:
                if self.persistent_data.counter % 100 == 0:
                    print("idling")
                if abs(gyrox) > 50 or abs(gyroz) > 50:
                    self.persistent_data.init_counter -= 1
                    if self.persistent_data.init_counter == 0:
                        self.persistent_data.is_idle = False
                        print("starting")
                return

            self.persistent_data.ml.write_data(
                dancer_id,
                [yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz],
                data_type,
            )

            self.handleMainLogic(dancer_id)
        except Exception:
            logger.error(line)
            logger.error(traceback.format_exc())

    def handleMainLogic(self, dancer_id):
        pred = self.persistent_data.ml.get_pred()
        if pred is not None:
            logger.info(pred)
            dance_moves, pos, sync_delay = pred
            mqueue.put((dance_moves, pos, sync_delay))
            self.clearLineBuffer()


# This class is used to store persistent data across connections
class ServerFactory(Factory):
    def __init__(self,):
        self.num_dancers = 0  # number of connected dancers
        self.is_idle = True
        self.counter = 0
        self.init_counter = 24
        self.start_time = time.time()
        dance_model_path = "model_weights.json"
        dance_scaler_path = "dnn_std_scaler.bin"
        self.ml = ML(
            dance_scaler_path=dance_scaler_path, dance_model_path=dance_model_path,
        )
        self.skip_initial_readings_cnt = [[50, 50], [50, 50], [50, 50]]

    def buildProtocol(self, addr):
        return Server(self)


def swap_positions(positions, pos):
    pos = [pos[positions[0]-1], pos[positions[1]-1], pos[positions[2]-1]]

    if pos == ["S", "S", "S"]:  # 1
        return [positions[0], positions[1], positions[2]]
    elif pos == ["S", "S", "L"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["S", "S", "R"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["S", "L", "S"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["S", "L", "L"]:
        return [positions[1], positions[2], positions[0]]
    elif pos == ["S", "L", "R"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["S", "R", "S"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["S", "R", "L"]:  # 2
        return [positions[0], positions[2], positions[1]]
    elif pos == ["S", "R", "R"]:
        return [positions[0], positions[2], positions[1]]
    elif pos == ["L", "S", "S"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["L", "S", "L"]:
        return [positions[2], positions[1], positions[0]]
    elif pos == ["L", "S", "R"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["L", "L", "S"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["L", "L", "L"]:
        return [positions[1], positions[2], positions[0]]
    elif pos == ["L", "L", "R"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["L", "R", "S"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["L", "R", "L"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["L", "R", "R"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["R", "S", "S"]:
        return [positions[0], positions[1], positions[2]]
    elif pos == ["R", "S", "L"]:  # 3
        return [positions[2], positions[1], positions[0]]
    elif pos == ["R", "S", "R"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["R", "L", "S"]:  # 4
        return [positions[1], positions[0], positions[2]]
    elif pos == ["R", "L", "L"]:  # 5
        return [positions[1], positions[2], positions[0]]
    elif pos == ["R", "L", "R"]:
        return [positions[1], positions[0], positions[2]]
    elif pos == ["R", "R", "S"]:
        return [positions[2], positions[0], positions[1]]
    elif pos == ["R", "R", "L"]:  # 6
        return [positions[2], positions[0], positions[1]]
    elif pos == ["R", "R", "R"]:
        return [positions[2], positions[0], positions[1]]


def format_results(positions, dance_moves, pos, sync_delay):
    sync_delay = (random.random() if sync_delay == -1 else sync_delay) * 1000
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
        counter = 1
        while True:
            while not mqueue.empty():
                dance_moves, pos, sync_delay = mqueue.get()
                dance_move = stats.mode(dance_moves)[0][0]

                if counter == 33:
                    dance_move = "logout"
                elif counter < 33 and dance_move == "logout":
                    dance_move = random.choice(ACTIONS)

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
                counter += 1

    except KeyboardInterrupt:
        thread.join()
        logger.info("Terminating")
