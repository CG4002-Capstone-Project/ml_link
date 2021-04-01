# The server

# NOTE
# ====
# If we are using SSH port forwarding to communicate between the
# laptop and the server, there is no need to encrypt/decrypt with
# AES as the data is already strongly encrypted.
#
# Reference: https://blog.eccouncil.org/what-is-ssh-port-forwarding/

import collections
import logging
import os
import random
import sys
import threading
import time
import traceback

from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver

from eval_client import ACTIONS, POSITIONS, Client

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
            gyrx, gyry, gyrz, accx, accy, accz, emg, dancer_id = line[1:].split(",")

            # appends data for each dancer to window
            dancer_id = int(dancer_id)
            gyrx, gyry, gyrz, accx, accy, accz = (
                float(gyrx),
                float(gyry),
                float(gyrz),
                float(accx),
                float(accy),
                float(accz),
            )
            self.persistent_data.data[dancer_id].append(
                [gyrx, gyry, gyrz, accx, accy, accz]
            )

            self.handleMainLogic(dancer_id)
        except Exception:
            logger.error(line)
            logger.error(traceback.format_exc())

    def handleMainLogic(self, dancer_id):
        # starts evaluation
        if not all(self.persistent_data.is_start):
            self.handleStart(dancer_id)
            return

        # transitions after display changes for 1.5s
        if self.persistent_data.is_transition_timer:
            self.persistent_data.is_transition_timer = False
            self.setTimerDelay(1.5, self.endTransition)
            return

        if self.persistent_data.is_transition:
            return

        # moves to next positions within 1.5s
        if self.persistent_data.is_position_changing_timer:
            self.persistent_data.is_position_changing_timer = False
            self.setTimerDelay(1.5, self.endPositionChanging)
            return

        if self.persistent_data.is_position_changing:
            return

        # dances for 5s
        if self.persistent_data.is_dancing_timer:
            self.persistent_data.is_dancing_timer = False
            self.setTimerDelay(5, self.endDancing)
            return

        if self.persistent_data.is_dancing:
            return

        if self.persistent_data.dance_moves[dancer_id] is None:
            self.handleDanceMoves(dancer_id)
            return

        # send to evaluation
        if all(self.persistent_data.dance_moves):
            self.endEvaluation()
            self.reset()
            return

        return

    def handleStart(self, dancer_id):
        # TODO: replaces with spike detection
        if not self.persistent_data.is_start[dancer_id]:
            is_spike = len(self.persistent_data.data[dancer_id]) == WINDOW_SIZE
            if is_spike:
                self.persistent_data.is_start[dancer_id] = True

        if all(self.persistent_data.is_start):
            self.startEvaluation()

    def endTransition(self):
        self.persistent_data.is_transition = False

    def endPositionChanging(self):
        self.persistent_data.is_position_changing = False
        self.persistent_data.positions = self.handlePositions()

    def handlePositions(self):
        return random.choice(POSITIONS)

    def handleSyncDelay(self):
        return random.random()

    def handleDanceMoves(self, dancer_id):
        self.persistent_data.dance_moves[dancer_id] = random.choice(ACTIONS)

    def endDancing(self):
        self.persistent_data.is_dancing = False

    def setTimerDelay(self, delay, func):
        self.persistent_data.timer = threading.Timer(delay, func)
        self.persistent_data.timer.start()

    def startEvaluation(self):
        self.persistent_data.my_client.send_message(
            "1 2 3" + "|" + "start" + "|" + "1.5" + "|"
        )

    def endEvaluation(self):
        sync_delay = self.handleSyncDelay()
        eval_data = f"1 2 3|{self.persistent_data.dance_moves[0]}|{sync_delay}|"
        self.persistent_data.my_client.send_message(eval_data)
        logger.info(f"sending to eval: {eval_data}")

    def reset(self):
        self.persistent_data.dance_moves = [
            None,
        ]  # [None, None, None]

        # timer for transition
        self.persistent_data.is_transition = True
        self.persistent_data.is_transition_timer = True

        # timer for changing position
        self.persistent_data.is_position_changing = True
        self.persistent_data.is_position_changing_timer = True

        # timer for dancing
        self.persistent_data.is_dancing = True
        self.persistent_data.is_dancing_timer = True

        self.persistent_data.positions = (
            self.persistent_data.my_client.receive_dancer_position()
        )
        logger.info(f"received positions: {self.persistent_data.positions}")


# This class is used to store persistent data across connections
class ServerFactory(Factory):
    def __init__(
        self, group_id="18", secret_key="1234123412341234",
    ):
        self.num_dancers = 0  # number of connected dancers
        self.my_client = Client(IP_ADDRESS, EVAL_PORT, group_id, secret_key)
        self.data = [
            collections.deque([], WINDOW_SIZE),
            collections.deque([], WINDOW_SIZE),
            collections.deque([], WINDOW_SIZE),
        ]
        self.is_start = [
            False,
        ]  # [False, False, False]
        self.dance_moves = [
            None,
        ]  # [None, None, None]
        self.positions = "1 2 3"

        self.timer = None

        # timer for transition
        self.is_transition = True
        self.is_transition_timer = True

        # timer for changing position
        self.is_position_changing = True
        self.is_position_changing_timer = True

        # timer for dancing
        self.is_dancing = True
        self.is_dancing_timer = True

    def buildProtocol(self, addr):
        return Server(self)


if __name__ == "__main__":
    logger.info("Started server on port %d" % DANCE_PORT)
    try:
        reactor.listenTCP(DANCE_PORT, ServerFactory())
        reactor.run()
    except KeyboardInterrupt:
        logger.info("Terminating")
