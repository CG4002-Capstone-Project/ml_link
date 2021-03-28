import argparse
import time
import traceback

import serial

# from datacollector import DataCollector

# SERIAL_PORT = os.environ['SERIAL_PORT']
SERIAL_PORTS = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]


def checkIMU(line):
    parsedline = ""
    try:
        gyrx, gyry, gyrz, accx, accy, accz, cksum = line.split(" ")
        val = int(gyrx) ^ int(gyry) ^ int(gyrz) ^ int(accx) ^ int(accy) ^ int(accz)
        if val == int(cksum):
            parsedline = (
                "#"
                + gyrx
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
            return parsedline
        return parsedline
    except Exception:
        print(traceback.format_exc())
        return parsedline


def checkEMG(line):
    parsedline = ""
    try:
        mav, rms, freq, cksum = line.split(" ")
        val = int(mav) ^ int(rms) ^ int(freq)
        if val == int(cksum):
            parsedline = "$" + mav + " " + rms + " " + freq
            return parsedline
        return parsedline
    except Exception:
        print(traceback.format_exc())
        return parsedline


class IntComm:
    def __init__(self, serial_port, dancer=1):
        self.ser = serial.Serial(SERIAL_PORTS[serial_port], 115200, timeout=0.5)
        self.ser.flushInput()
        time.sleep(0.5)
        print("Opened serial port %s" % SERIAL_PORTS[serial_port])

        while True:
            line = self.ser.readline()
            print(line)
            if b"Send any character to begin DMP programming and demo:" in line:
                break

        self.ser.write("s".encode())

        self.dancer = dancer

    def get_line(self):

        # Get data from beetles and if not reconnect
        line = ""
        count = 0
        while True:
            line = self.ser.readline()
            if len(line) > 0:
                line = line.decode().strip()
                break
            else:
                count += 1
                print("no data received for {} iterations".format(count))
                if count >= 5:
                    count = 0
                    print("reconnecting with beetle...")
                    self.ser.write("s".encode())

        try:
            # handshake message in the case of reconnection
            if "Send any character to begin DMP programming and demo:" in line:
                self.ser.write("s".encode())
                return self.get_line()

            # initial messages to be ignored
            if line[0] == "!":
                print("data to be ignored")
                print(line[1:])
                self.ser.write("s".encode())
                return self.get_line()

            # EMG messages
            if line[0] == "$":
                parsedline = checkEMG(line[1:])
                if parsedline == "":
                    print("checksum failed for EMG data")
                    return self.get_line()
                else:
                    print(parsedline)
                    return parsedline

            # acc/gyr data messages
            if line[0] == "#":
                parsedline = checkIMU(line[1:])
                if parsedline == "":
                    print("checksum failed for IMU data")
                    return self.get_line()
                else:
                    # print(parsedline)
                    return parsedline

            print("Invalid message")
            print(line)

            return self.get_line()
        except Exception:
            print(traceback.format_exc())
            return self.get_line()

    # helper function to get raw data
    def get_acc_gyr_data(self):
        line = self.get_line()
        # only take in IMU data
        if line[0] == "#":
            tokens = line[1:].split(" ")
            # print (tokens)
            data = [float(token) for token in tokens]
            return data
        else:
            return self.get_acc_gyr_data()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument(
        "--serial", default=0, help="select serial port", type=int, required=True
    )
    args = parser.parse_args()
    serial_port_entered = args.serial
    int_comm = IntComm(serial_port_entered)

    count = 0
    start = time.time()

    while True:
        line = int_comm.get_line()
        count = count + 1

        print(line)

        if count % 150 == 0:
            end = time.time()
            print("Receiving data at %d Hz" % round(150 / (end - start)))
            start = time.time()
