import serial
import os
import time

SERIAL_PORT = os.environ['SERIAL_PORT'] if 'SERIAL_PORT' in os.environ else 'No Serial Port Given'


class FineFood():
    def __init__(self, serial_port):
        self.ser = serial.Serial(serial_port, 115200)
        self.ser.flushInput()
        print("Opened serial port %s" % serial_port)

    def get_line(self):
        line = self.ser.readline()
        if len(line) > 0:
            line = line.decode().strip()
        else:
            return self.get_line()

        while (len(line) > 0 and line[0] != '#'):
            line = self.ser.readline().decode().strip()
            print("Invalid message")

        return line[1:]

    def get_acc_gyr_data(self):
        line = self.get_line()
        tokens = line.split(" ")[2:]
        return [float(token) for token in tokens]

if __name__ == "__main__":
    fine_food = FineFood(SERIAL_PORT)

    count = 0
    start = time.time()

    while True:
        line = fine_food.get_line()
        count = count + 1

        print(line)

        if count % 150 == 0:
            end = time.time()
            print("Receiving data at %d Hz" % round(150/(end-start)))
            start = time.time()


