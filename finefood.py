import serial
import os
import time

SERIAL_PORT = os.environ['SERIAL_PORT'] if 'SERIAL_PORT' in os.environ else 'No Serial Port Given'


class FineFood():
    def __init__(self, serial_port):
        self.ser = serial.Serial(serial_port, 115200)
        self.ser.flushInput()
        print("Opened serial port %s" % serial_port)

        # Initialization routine
        line = self.ser.readline()
        print(line)
        line = self.ser.readline()
        print(line)

        self.ser.write("s".encode())

        for _ in range(4):
            line = self.ser.readline()
            print(line)


    def get_line(self):
        line = self.ser.readline()
        if len(line) > 0:
            line = line.decode().strip()
        else:
            return self.get_line()

        # status messages; print and get another line
        if line[0] == '!':
            print(line[1:])
            return self.get_line()

        # acc/gyr data messages
        if line[0] == '#':
            return line[1:]

        print("Invalid message")
        print(line)

        return self.get_line()

    def get_acc_gyr_data(self):
        line = self.get_line()
        tokens = line.split(" ")
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


