# The client file that is used from the Ultra96 to the evaluation server connection
# For now the client will only send certain format of message.

import base64
import os
import socket
import sys
import threading
import time

from Crypto import Random
from Crypto.Cipher import AES

# Week 13 test: 8 moves, so 33 in total = (8*4) + 1 (logout)
ACTIONS = [
    "gun",
    "sidepump",
    "hair",
    "pointhigh",
    "elbowkick",
    "listen",
    "dab",
    "wipetable",
]
POSITIONS = ["1 2 3", "3 2 1", "2 3 1", "3 1 2", "1 3 2", "2 1 3"]
LOG_DIR = os.path.join(os.path.dirname(__file__), "evaluation_logs")
NUM_MOVE_PER_ACTION = 4
N_TRANSITIONS = 6
MESSAGE_SIZE = 3  # position, 1 action, sync

ENCRYPT_BLOCK_SIZE = 16


class Client(threading.Thread):
    def __init__(self, ip_addr, port_num, group_id, key):
        super(Client, self).__init__()

        self.idx = 0
        self.timeout = 60
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False

        self.group_id = group_id
        self.key = key

        self.dancer_positions = ["1", "2", "3"]

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print(
            "Start connecting... server address: %s port: %s" % server_address,
            file=sys.stderr,
        )
        self.socket.connect(server_address)
        print("Connected")

    # To encrypt the message, which is a string
    def encrypt_message(self, message):
        raw_message = "#" + message
        padded_raw_message = raw_message + " " * (
            ENCRYPT_BLOCK_SIZE - (len(raw_message) % ENCRYPT_BLOCK_SIZE)
        )
        iv = Random.new().read(AES.block_size)
        secret_key = bytes(str(self.key), encoding="utf8")
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        encrypted_message = base64.b64encode(
            iv + cipher.encrypt(bytes(padded_raw_message, "utf8"))
        )
        return encrypted_message

    # To send the message to the sever
    def send_message(self, message):
        encrypted_message = self.encrypt_message(message)
        self.socket.sendall(encrypted_message)

    def receive_dancer_position(self):
        dancer_position = self.socket.recv(1024)
        msg = dancer_position.decode("utf8")
        return msg

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()


def main():
    ip_addr = "localhost"
    port_num = 8000
    group_id = "18"
    key = "1234123412341234"

    my_client = Client(ip_addr, port_num, group_id, key)

    index = 0

    time.sleep(15)

    while True:
        if index == 0:
            my_client.send_message("1 2 3" + "|" + "start" + "|" + "1.5" + "|")
        dancer_position = my_client.receive_dancer_position()
        print("dancer_position: " + dancer_position)
        my_client.send_message("1 2 3" + "|" + "muscle" + "|" + "1.5" + "|")

        print("Received dancer postions: ", str(dancer_position))
        time.sleep(4)
        index += 1
        if index == 60:
            my_client.send_message(dancer_position + "|" + "logout" + "|" + "1.5" + "|")
            my_client.stop()


if __name__ == "__main__":
    main()
