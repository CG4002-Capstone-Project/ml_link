# Main and only program to run on each laptop

# NOTE
# ====
# If we are using SSH port forwarding to communicate between the
# laptop and the server, there is no need to encrypt/decrypt with
# AES as the data is already strongly encrypted.
#
# Reference: https://blog.eccouncil.org/what-is-ssh-port-forwarding/

import logging
import os
import socket
import sys
import time
import traceback

from intcomm import IntComm

PORT = int(os.environ["DANCE_PORT"])
DANCER_ID = int(os.environ["DANCER_ID"])
HOST = "localhost"


# setup logging
file_handler = logging.FileHandler(
    filename=f'logs/laptop_{DANCER_ID}_{time.strftime("%Y%m%d-%H%M%S")}.log'
)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("ultra96")


class Laptop:
    def __init__(self):
        self.intcomm = IntComm(0, DANCER_ID)
        self.buffer = []

    def collect_data(self):
        # #yaw,pitch,roll,accx,accy,accz,emg
        data = self.intcomm.get_line()
        try:
            if len(data) == 0 or data[0] != "#":
                logger.error("Invalid data:", data)
                raise "Invalid data"

            data = f"#{DANCER_ID},{data[1:]}\n"
            logger.info(data)
            self.buffer.append(data)
        except:
            logger.error(data)
            logger.error(traceback.print_exc())
            return self.collect_data()

    def send_data(self, sock):
        if len(self.buffer) == 5:
            line = "".join(self.buffer)
            sock.sendall(line.encode())
            self.buffer = []

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))

            while True:
                try:
                    self.collect_data()
                    self.send_data(sock)
                except Exception:
                    logger.error(traceback.format_exc())


if __name__ == "__main__":
    laptop = Laptop()
    laptop.run()
