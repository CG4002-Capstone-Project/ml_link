import argparse
import time

import serial

SERIAL_PORTS = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]


class IntComm:
    def __init__(self, serial_port):
        self.ser = serial.Serial(SERIAL_PORTS[serial_port], 115200, timeout=0.5)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.flush()
        print("Opened serial port %s" % SERIAL_PORTS[serial_port])

    def get_line(self):
        ln = self.ser.readline().decode().strip()
        idx = ln.rfind("#")
        return ln[idx:]


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument(
        "--serial", default=0, help="select serial port", type=int, required=True
    )
    args = parser.parse_args()
    serial_port_entered = args.serial

    # Initialise intcomm
    int_comm = IntComm(serial_port_entered)

    count = 0
    start = time.time()

    while True:
        line = int_comm.get_line()
        count = count + 1

        print(line)

        if count % 100 == 0:
            end = time.time()
            print("Receiving data at %f Hz" % (100 / (end - start)))
            start = time.time()
