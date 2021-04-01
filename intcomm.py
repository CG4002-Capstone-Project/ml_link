import argparse
import time
import traceback

import serial

SERIAL_PORTS = ["/dev/cu.usbmodem14201", "/dev/ttyACM1", "/dev/ttyACM2"]


class IntComm:
    def __init__(self, serial_port, dancer=1):
        self.dancer = dancer
        self.ser = serial.Serial(SERIAL_PORTS[serial_port], 115200, timeout=0.5)
        self.ser.flushInput()
        print("Opened serial port %s" % SERIAL_PORTS[serial_port])

    def get_line(self):
        try:
            line = self.ser.readline().decode().strip()
            if len(line) == 0 or line[0] != "#":
                print("Invalid line:", line)
                return self.get_line()
            return line
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return self.get_line()


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
