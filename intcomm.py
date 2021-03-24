import serial
import os
import time
#from datacollector import DataCollector

#SERIAL_PORT = os.environ['SERIAL_PORT']
SERIAL_PORT = '/dev/ttyACM0'

def check(line):
        try:
            gyrx, gyry, gyz, accx, accy, accz, cksum = line.split(" ")
            val = int(gyrx) ^ int(gyry) ^ int(gyz) ^ int(accx) ^ int(accy) ^ int(accz)    
            newline = ""
            if (val == int(cksum)):
                newline = gyrx + " " + gyry + " " + gyz + " " + accx + " " + accy + " " + accz
            return newline                
        except Exception as e:
            print (e)
            return newline


class IntComm():
    def __init__(self, serial_port, dancer = 1):
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

        # for dashboard
        #self.datacollector = DataCollector("localhost", 8086, "admin", "xilinx123")
        self.dancer = dancer

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
            newline = check(line[1:])
            if (newline != ""):
                print ("checksum passed")
                return newline
            else:
                print ("checksum failed")
                return self.get_line()

        print("Invalid message")
        print(line)

        return self.get_line()


    # helper function when you don't need any other data (they are ignored)
    def get_acc_gyr_data(self):
        line = self.get_line()
        tokens = line.split(" ")

        data = [float(token) for token in tokens]
        #self.datacollector.insert_gyr_data(int(time.time()), self.dancer, data[0], data[1], data[2])
        #self.datacollector.insert_acc_data(int(time.time()), self.dancer, data[3], data[4], data[5])

        return data



if __name__ == "__main__":
    int_comm = IntComm(SERIAL_PORT)

    count = 0
    start = time.time()

    while True:
        line = int_comm.get_line()
        count = count + 1

        print(line)

        if count % 150 == 0:
            end = time.time()
            print("Receiving data at %d Hz" % round(150/(end-start)))
            start = time.time()
