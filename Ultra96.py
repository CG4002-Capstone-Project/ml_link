# The server

# NOTE
# ====
# If we are using SSH port forwarding to communicate between the
# laptop and the server, there is no need to encrypt/decrypt with
# AES as the data is already strongly encrypted.
#
# Reference: https://blog.eccouncil.org/what-is-ssh-port-forwarding/

import os

from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver

PORT = os.environ.get("SERIAL_PORT", 8000)


class Server(LineReceiver):
    delimiter = b"\n"

    def __init__(self, persistent_data):
        self.persistent_data = persistent_data

    def connectionMade(self):
        print("New dancer")
        self.persistent_data.num_dancers += 1
        self.print_num_dancers()

    def connectionLost(self, reason):
        print("A dancer disconnected")
        self.persistent_data.num_dancers -= 1
        self.print_num_dancers()

    def print_num_dancers(self):
        print(
            "There are currently %d connected dancers."
            % self.persistent_data.num_dancers
        )

    def lineReceived(self, line):
        line = line.decode()
        if line[0] != "#":
            print("Received invalid data", line)
            return
        print(line)


# This class is used to store persistent data across connections
class ServerFactory(Factory):
    def __init__(self):
        self.num_dancers = 0  # number of connected dancers

    def buildProtocol(self, addr):
        return Server(self)


if __name__ == "__main__":
    print("Started server on port %d" % PORT)
    reactor.listenTCP(PORT, ServerFactory())
    reactor.run()
