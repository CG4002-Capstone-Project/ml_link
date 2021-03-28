import time
import traceback
import argparse
import serial
from bluepy import btle

# from datacollector import DataCollector

# SERIAL_PORT = os.environ['SERIAL_PORT']
#SERIAL_PORTS = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]
MAC = ["80:30:DC:E9:25:07", "34:B1:F7:D2:35:97", "34:B1:F7:D2:35:9D"]

def checkIMU(line):
    parsedline = ""
    try:
        gyrx, gyry, gyrz, accx, accy, accz, cksum = line.split(" ")
        val = int(gyrx) ^ int(gyry) ^ int(gyrz) ^ int(accx) ^ int(accy) ^ int(accz)
        if val == int(cksum):
            parsedline = "#" + gyrx + " " + gyry + " " + gyrz + " " + accx + " " + accy + " " + accz
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

class UUIDS:
    SERIAL_COMMS = btle.UUID("0000dfb1-0000-1000-8000-00805f9b34fb")

class Delegate(btle.DefaultDelegate):
    def __init__(self, params):
        btle.DefaultDelegate.__init__(self)

    def handleNotification(self, cHandle, data):
        print (data)

class IntComm:
    def __init__(self, port, dancer=1):
        
        self.global_delegate_obj = 0
        self.global_beetle = 0
        self.address = MAC[port] 
        
        # establish connection
        self.establish_connection(self.address)
          
    def establish_connection(self, address):
        while True:
            try:
                if self.global_beetle != 0:  # disconnect before reconnect
                    self.global_beetle._stopHelper()
                    self.global_beetle.disconnect()
                    self.global_beetle = 0
                        
                if self.global_beetle == 0:  # just stick with if instead of else
                    print("connecting with %s" % (address))
                    # creates a Peripheral object and makes a connection to the device
                    beetle = btle.Peripheral(address)
                    self.global_beetle = beetle
                    # creates and initialises the object instance.
                    beetle_delegate = Delegate(address)
                    self.global_delegate_obj = beetle_delegate
                    # stores a reference to a “delegate” object, which is called when asynchronous events such as Bluetooth notifications occur.
                    beetle.withDelegate(beetle_delegate)
                    
                    # do handshake
                    while True:
                        for characteristic in beetle.getCharacteristics():
                            if characteristic.uuid == UUIDS.SERIAL_COMMS:
                                print("sending 'H' packet to %s" % (beetle.addr))
                                characteristic.write(bytes("H", "UTF-8"), withResponse=False)

                                if beetle.waitForNotifications(5):
                                    print ("connected....")
                                    break
                                else:
                                    print("sending 'H' packet to %s" % (beetle.addr))
                                    characteristic.write(bytes("H", "UTF-8"), withResponse=False)
                           
            except KeyboardInterrupt:
                print(traceback.format_exc())
                if self.global_beetle != 0:  # disconnect
                    self.global_beetle._stopHelper()
                    self.global_beetle.disconnect()
                    self.global_beetle = 0
                sys.exit()
            except Exception:
                print(traceback.format_exc())
                self.establish_connection(address)
        
    def get_line(self):
        
        waitCount = 0
        while True:
            try:
                if self.global_beetle.waitForNotifications(2):
                    return
                else:
                    waitCount += 1
                    if waitCount >= 10:
                        waitCount = 0
                        self.establish_connection(self.address)
                        return

            except KeyboardInterrupt:
                print(traceback.format_exc())
                if global_beetle[0] != 0:  # disconnect
                    global_beetle[0]._stopHelper()
                    global_beetle[0].disconnect()
                    global_beetle[0] = 0
                sys.exit()
            except Exception:
                print(traceback.format_exc())
                establish_connection(beetle.addr)
                return


        """
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
                print ("no data received for {} iterations".format(count))
                if (count >= 5):
                    count = 0
                    print ("reconnecting with beetle...")
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
                if (parsedline == ""):
                    print("checksum failed for EMG data")
                    return self.get_line()
                else:
                    print(parsedline)
                    return parsedline

            # acc/gyr data messages
            if line[0] == "#":
                parsedline = checkIMU(line[1:])
                if (parsedline == ""):
                    print("checksum failed for IMU data")
                    return self.get_line()
                else:
                    print(parsedline)
                    return parsedline

            print("Invalid message")
            print(line)

            return self.get_line()
        except Exception:
            print(traceback.format_exc())
            return self.get_line()
        """

    # helper function to get raw data
    def get_acc_gyr_data(self):
        line = self.get_line()
        # only take in IMU data
        if (line[0] == "#"):
            tokens = line[1:].split(" ")
            print (tokens)
            data = [float(token) for token in tokens]
            return data
        else:
            return self.get_acc_gyr_data()
      

if __name__ == "__main__":
    
    int_comm = IntComm(1)

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
