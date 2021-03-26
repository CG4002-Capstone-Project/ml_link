import argparse
import base64
import socket
import threading
import time

from Crypto import Random
from Crypto.Cipher import AES

from intcomm import SERIAL_PORT_0, SERIAL_PORT_1, IntComm

PORT = 9092
DANCER_ID = 1
HOST = "localhost"
PORT_NUM = [9091, 9092, 9093]
ENCRYPT_BLOCK_SIZE = 16


class Client(threading.Thread):
    def __init__(self, group_id, key):

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

        serial_port = SERIAL_PORT_0
        if dancer_id == 1:
            serial_port = SERIAL_PORT_1

        self.intcomm = IntComm(serial_port, dancer_id)

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        print("Start connecting>>>>>>>>>>>>")
        self.socket.connect(server_address)
        print("Connected")

    def run(self):

        RTT = 0.0
        offset = 0.0
        count = 0
        start = time.time()
        t1 = time.time()

        while True:
            if count % 10 == 0:
                end = time.time()
                print("Receiving data at %d Hz" % round(150 / (end - start)))
                start = time.time()
            count += 1

            message_final = (
                str(dancer_id)
                + "|"
                + str(t1)
                + "|"
                + str(offset)
                + "|"
                + self.intcomm.get_line()
                + "|"
            )
            print("Sending", message_final)

            t1 = time.time()
            self.send_message(message_final)
            timestamp = self.receive_timestamp()
            t4 = time.time()
            t2 = float(timestamp.split("|")[0][:18])
            t3 = float(timestamp.split("|")[1][:18])
            RTT = t4 - t3 + t2 - t1
            offset = (t2 - t1) - RTT / 2

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

    def receive_timestamp(self):
        timestamp = self.socket.recv(1024)
        msg = timestamp.decode("utf8")
        return msg

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument("--dancer_id", help="dancer id", type=int, required=True)

    args = parser.parse_args()
    dancer_id = args.dancer_id

    ip_addr = "127.0.0.1"
    port_num = PORT_NUM[dancer_id]
    group_id = "18"
    key = "1234123412341234"
    my_client = Client(group_id, key)

    my_client.run()
