import argparse
import base64
import socket
import threading
import time

import pika
from Crypto import Random
from Crypto.Cipher import AES

from intcomm import SERIAL_PORT_0, SERIAL_PORT_1, IntComm

PORT = 9092
DANCER_ID = 1
HOST = "localhost"
PORT_NUM = [9091, 9092, 9093]
DB_QUEUES = ["trainee_one_data", "trainee_two_data", "trainee_three_data"]
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
            data = self.intcomm.get_line()
            t1 = time.time()
            # IMU data
            if data[0] == "#":
                data = data[1:]
                gyrx, gyry, gyrz, accx, accy, accz, cksum = data.split(" ")
                data = (
                    gyrx
                    + " "
                    + gyry
                    + " "
                    + gyrz
                    + " "
                    + accx
                    + " "
                    + accy
                    + " "
                    + accz
                )
                message_final = (
                    str(dancer_id)
                    + "|"
                    + str(t1)
                    + "|"
                    + str(offset)
                    + "|"
                    + data
                    + "|"
                )
                database_msg = str(dancer_id) + "|" + str(t1) + "|" + data + "|"
                channel.basic_publish(
                    exchange="", routing_key=DB_QUEUES[dancer_id], body=database_msg
                )
                print("Sending IMU", message_final)
                t1 = time.time()

                self.send_message(message_final)
                timestamp = self.receive_timestamp()
                t4 = time.time()
                t2 = float(timestamp.split("|")[0][:18])
                t3 = float(timestamp.split("|")[1][:18])
                RTT = t4 - t3 + t2 - t1
                offset = (t2 - t1) - RTT / 2

            # EMG data
            elif data[0] == "$":
                if emg is True:
                    data = data[1:]
                    mav, rms, freq, cksum = data.split(" ")
                    data = mav + " " + rms + " " + freq
                    message_emg = str(t1) + "|" + data
                    print("Sending EMG", message_emg)
                    channel.basic_publish(
                        exchange="", routing_key="emg", body=message_emg
                    )

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


# database_msg = "1|time.time()|[a,b,c,d,e,f]|"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="External Comms")
    parser.add_argument("--dancer_id", help="dancer id", type=int, required=True)
    parser.add_argument("--emg", default=False, help="collect emg data", type=bool)

    args = parser.parse_args()
    dancer_id = args.dancer_id
    emg = args.emg

    CLOUDAMQP_URL = "amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu"
    params = pika.URLParameters(CLOUDAMQP_URL)
    params.socket_timeout = 5

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=DB_QUEUES[dancer_id])

    if emg is True:
        CLOUDAMQP_URL = "amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu"
        params = pika.URLParameters(CLOUDAMQP_URL)
        params.socket_timeout = 5

        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue="emg")

    ip_addr = "127.0.0.1"
    port_num = PORT_NUM[dancer_id]
    group_id = "18"
    key = "1234123412341234"
    my_client = Client(group_id, key)

    my_client.run()
