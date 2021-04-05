# The server

# NOTE
# ====
# If we are using SSH port forwarding to communicate between the
# laptop and the server, there is no need to encrypt/decrypt with
# AES as the data is already strongly encrypted.
#
# Reference: https://blog.eccouncil.org/what-is-ssh-port-forwarding/

import logging
import os
import random
import sys
import time
import traceback

from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver

from eval_client import Client
from ML import ML

IP_ADDRESS = os.environ["IP_ADDRESS"]
EVAL_PORT = int(os.environ["EVAL_PORT"])
DANCE_PORT = int(os.environ["DANCE_PORT"])

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

            self.persistent_data.ml.write_data(
                dancer_id, [yaw, pitch, roll, accx, accy, accz]
            )

            self.handleMainLogic(dancer_id)
        except Exception:
            logger.error(line)
            logger.error(traceback.format_exc())

    def handleMainLogic(self, dancer_id):
        # return pred => (dance_move: str, dance_positions: list[int, int, int]) or None
        pred = self.persistent_data.ml.get_pred()
        if pred is not None:
            dance_move, dance_positions = pred
            self.persistent_data.endEvaluation(dance_move, dance_positions)
            self.persistent_data.reset()


# This class is used to store persistent data across connections
class ServerFactory(Factory):
    def __init__(
        self, group_id="18", secret_key="1234123412341234",
    ):
        self.num_dancers = 0  # number of connected dancers
        self.my_client = Client(IP_ADDRESS, EVAL_PORT, group_id, secret_key)

        self.ml = ML(
            on_fpga=False,
            dance_scaler_path="./dance_scaler.bin",
            pos_scaler_path="./pos_scaler.bin",
            dance_model_path="./dance_model.pth",
            pos_model_path="./pos_model.pth",
        )

    def buildProtocol(self, addr):
        return Server(self)

    def endEvaluation(self, dance_move, dance_positions):
        sync_delay = self.handleSyncDelay()
        dance_positions = " ".join(
            [str(dance_position) for dance_position in dance_positions]
        )
        eval_data = f"{dance_positions}|{dance_move}|{sync_delay}|"
        self.my_client.send_message(eval_data)
        logger.info(f"sending to eval: {eval_data}")

    def reset(self):
        # timer for transition
        self.is_transition = True
        self.is_transition_timer = True

        dance_positions = self.my_client.receive_dancer_position()
        logger.info(f"received positions: {dance_positions}")
        dance_positions = [
            int(dance_position) for dance_position in dance_positions.split(" ")
        ]
        self.ml.set_pos(dance_positions)

    def handleSyncDelay(self):
        return random.random()


if __name__ == "__main__":
    logger.info("Started server on port %d" % DANCE_PORT)
    try:
        reactor.listenTCP(DANCE_PORT, ServerFactory())
        reactor.run()
    except KeyboardInterrupt:
        logger.info("Terminating")
