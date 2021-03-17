from finefood import FineFood, SERIAL_PORT

class Deck():
    def __init__(self, serial_port):
        self.finefood = FineFood(serial_port)


if __name__ == "__main__":
    deck = Deck(SERIAL_PORT)

    while True:
        line = deck.finefood.get_acc_gyr_data()
        print(line)

