import time
import traceback

import serial

# from datacollector import DataCollector

# SERIAL_PORT = os.environ['SERIAL_PORT']
SERIAL_PORT_0 = "/dev/ttyACM0"
SERIAL_PORT_1 = "/dev/ttyACM1"


def checkIMU(line):
    try:
        gyrx, gyry, gyrz, accx, accy, accz, cksum = line.split(" ")
        val = int(gyrx) ^ int(gyry) ^ int(gyrz) ^ int(accx) ^ int(accy) ^ int(accz)
        if val == int(cksum):
            return True
        return False
    except Exception:
        print(traceback.format_exc())
        return False


def checkEMG(line):
    try:
        mav, rms, freq, cksum = line.split(" ")
        val = int(mav) ^ int(rms) ^ int(freq)
        if val == int(cksum):
            return True
        return False
    except Exception:
        print(traceback.format_exc())
        return False


class IntComm:
    def __init__(self, serial_port, dancer=1):
        self.ser = serial.Serial(serial_port, 115200)
        self.ser.flushInput()
        time.sleep(0.5)
        print("Opened serial port %s" % serial_port)

        while True:
            line = self.ser.readline()
            print(line)
            if b"Send any character to begin DMP programming and demo:" in line:
                break

        self.ser.write("s".encode())

        self.dancer = dancer

    def get_line(self):
        line = self.ser.readline()
        if len(line) > 0:
            line = line.decode().strip()
        else:
            return self.get_line()

        try:
            # initial messages to ignore
            if line[0] == "!":
                print("here")
                print(line[1:])
                return self.get_line()

            # EMG messages
            if line[0] == "$":
                if checkEMG(line[1:]) is True:
                    print(line)
                    return line
                else:
                    print("checksum failed for EMG data")
                    return self.get_line()

            # acc/gyr data messages
            if line[0] == "#":
                if checkIMU(line[1:]) is True:
                    print(line)
                    return line
                else:
                    print("checksum failed for IMU data")
                    return self.get_line()

            print("Invalid message")
            print(line)

            return self.get_line()
        except Exception:
            print(traceback.format_exc())
            return self.get_line()

    # helper function to get raw data
    def get_acc_gyr_data(self):
        line = self.get_line()
        tokens = line.split(" ")

        data = [float(token) for token in tokens]

        return data


if __name__ == "__main__":
    int_comm = IntComm(SERIAL_PORT_0)

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
