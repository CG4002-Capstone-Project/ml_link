# Main and only program to run on each laptop

# NOTE
# ====
# If we are using SSH port forwarding to communicate between the
# laptop and the server, there is no need to encrypt/decrypt with
# AES as the data is already strongly encrypted.
#
# Reference: https://blog.eccouncil.org/what-is-ssh-port-forwarding/

from intcomm import IntComm, SERIAL_PORT
import os
import socket

PORT = int(os.environ['DANCE_PORT'])
DANCER_ID = int(os.environ['DANCER_ID'])
HOST = 'localhost'

class Laptop():
    def __init__(self):
        self.intcomm = IntComm(SERIAL_PORT, DANCER_ID)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))

            while True:
                line = "#" + self.intcomm.get_line() + "\n"
                print("Sending", line)
                sock.sendall(line.encode())


if __name__ == "__main__":
    laptop = Laptop()
    laptop.run()
