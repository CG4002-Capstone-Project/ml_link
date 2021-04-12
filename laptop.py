import logging
import os
import socket
import sys
import time
import traceback

import pika

from intcomm import IntComm

PORT = int(os.environ["DANCE_PORT"])
DANCER_ID = int(os.environ["DANCER_ID"])
IS_DASHBOARD = bool(int(os.environ["IS_DASHBOARD"]))
IS_EMG = bool(int(os.environ["IS_EMG"]))
IS_POSITION = int(os.environ["IS_POSITION"])

HOST = "localhost"
DB_QUEUES = ["trainee_one_data", "trainee_two_data", "trainee_three_data"]


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
        self.intcomm = IntComm(IS_POSITION)
        self.buffer = []
        self.counter = 1
        self.start_time = time.time()

    def collect_data(self):
        # data: #yaw,pitch,roll,accx,accy,accz,emg
        try:
            data = self.intcomm.get_line()

            # display frequency
            self.counter += 1
            if self.counter % 100 == 0:
                end_time = time.time()
                logger.info(
                    "Receiving data at %f Hz" % (100 / (end_time - self.start_time))
                )
                self.start_time = end_time

            if len(data) == 0 or data[0] != "#":
                logger.error("Invalid data:", data)
                raise "Invalid data"

            formatted_data = f"#{DANCER_ID},{IS_POSITION},{data[1:]}\n"
            logger.debug(formatted_data)
            self.buffer.append(formatted_data)
            if IS_DASHBOARD:
                yaw, pitch, roll, gyrox, gyroy, gyroz, accx, accy, accz, emg = data[
                    1:
                ].split(",")
                imu_msg = " ".join([gyrox, gyroy, gyroz, accx, accy, accz])
                imu_msg = f"{str(DANCER_ID)}|{str(time.time())}|{imu_msg}|"
                logger.debug(f"imu_msg: {imu_msg}")
                channel.basic_publish(
                    exchange="", routing_key=DB_QUEUES[DANCER_ID], body=imu_msg,
                )

                if IS_EMG:
                    emg_msg = f"{str(time.time())}|{emg}"
                    logger.debug(f"emg_msg: {emg_msg}")
                    channel.basic_publish(exchange="", routing_key="emg", body=emg_msg)
        except KeyboardInterrupt:
            print("Terminating")
            sys.exit()
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
    if IS_DASHBOARD:
        CLOUDAMQP_URL = "amqps://yjxagmuu:9i_-oo9VNSh5w4DtBxOlB6KLLOMLWlgj@mustang.rmq.cloudamqp.com/yjxagmuu"
        params = pika.URLParameters(CLOUDAMQP_URL)
        params.socket_timeout = 5

        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=DB_QUEUES[DANCER_ID])
        if IS_EMG:
            channel.queue_declare(queue="emg")

    laptop = Laptop()
    laptop.run()
