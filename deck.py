from finefood import FineFood, SERIAL_PORT
from random import randrange
import time
import os

MOVES = [
        "dab",
        "elbowkick",
        "listen",
        "pointhigh",
        "hair",
        "gun",
        "sidepump",
        "wipetable",
        "logout",
        "idle"
]

FREQUENCY = 25
NUM_S_PER_MOVE = 25
NUM_MOVES = 12

def clr():
    if os.name == 'posix':
        _ = os.system('clear')
    else:
      _ = os.system('cls')

if __name__ == "__main__":
    print("Dance Data Collection Script\n"
          "============================\n")

    data = []


    # Get type
    line = ""
    while not (line == "m" or line == "s"):
        line = input("Do you want to record multiple moves(m) or just a single move(s)? ")

    moves = MOVES
    if line == "s":
        print("The following moves are available: ")
        for i, move in list(enumerate(moves)):
            print("%d: %s" % (i, move))
        print("")

        smove = -1
        while smove < 0 or smove >= len(moves):
            smove = int(input("Which move do you want to record? (0-%d) " % len(moves)))

        moves = [MOVES[smove]]

    timeout = 3
    while (timeout > 0):
        clr()
        print("Starting in %ds" % timeout)
        timeout = timeout - 1
        time.sleep(1)

    finefood = FineFood(SERIAL_PORT)

    for i in range(NUM_MOVES + 1):
        move = moves[randrange(len(moves))]

        clr()
        print("%d: %s" % (i, move if i > 0 else "Please wait"))

        data_c = 0
        while data_c < NUM_S_PER_MOVE:
            if i > 0:
                point = finefood.get_acc_gyr_data()
                point.append(move)
                data.append(point)
            else:
                finefood.get_line()

            data_c = data_c + 1

    for i, line in enumerate(data):
        data[i] = [str(x) for x in line]
    data = [",".join(line) for line in data]
    data = "\n".join(data) + "\n"

    line = ""
    while not (line == "y" or line == "n"):
        line = input("Do you want to keep the data? (y/n) ")

    if line == "y":
        f = open("data.csv", "a+")
        f.write(data)
        f.close()



