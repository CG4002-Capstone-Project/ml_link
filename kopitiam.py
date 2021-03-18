from influxdb import InfluxDBClient
import time
import math
import random
import traceback

class Kopitiam():
    def __init__(self, host, port, username, password):
        try:
            self.client = InfluxDBClient(host=host, port=port, username=username, password=password)

            # create database if not exist
            dbs = [x['name'] for x in self.client.get_list_database()]
            if 'rawdata' not in dbs:
                self.client.create_database('rawdata')
        except:
            self.client = None
            traceback.print_exc()

    def insert_raw_data(self, raw_data):
        try:
            if self.client is not None:
                self.client.switch_database('rawdata')
                self.client.write_points([raw_data])
        except:
            traceback.print_exc()

    def insert_acc_data(self, timestamp, dancer, accx, accy, accz):
        rd = {
            "measurement": "acc",
            "tags": {
                "dancer": str(dancer)
            },
            "timestamp": str(timestamp),
            "fields": {
                "x": accx,
                "y": accy,
                "z": accz
            }
        }
        self.insert_raw_data(rd)

    def insert_gyr_data(self, timestamp, dancer, yaw, pitch, roll):
        rd = {
            "measurement": "gyr",
            "tags": {
                "dancer": str(dancer)
            },
            "timestamp": str(timestamp),
            "fields": {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }
        }
        self.insert_raw_data(rd)


if __name__ == "__main__":
    kopitiam = Kopitiam("localhost", 8086, "admin", "xilinx123")

    for i in range(1000):
        print("inserting data")
        kopitiam.insert_acc_data(int(time.time()), 1, math.sin(i), math.sin(i + 3), math.sin(i + 6))
        kopitiam.insert_gyr_data(int(time.time()), 1, math.sin(i), math.sin(i + 3), math.sin(i + 6))
        time.sleep(0.1)

