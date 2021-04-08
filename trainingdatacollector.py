import argparse
import os
import time
from random import randrange

from intcomm import IntComm

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
]
positions = [
    "nothing",  # to simulate dancing immediately
    "right",
    "left",
    "rightright",
    "leftleft",
]

NUM_S_PER_MOVE = 150
NUM_MOVES = 25


def clr():
    if os.name == "posix":
        _ = os.system("clear")
    else:
        _ = os.system("cls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal Comms")
    parser.add_argument(
        "--serial", default=0, help="select serial port", type=int, required=True
    )
    args = parser.parse_args()
    serial_port_entered = args.serial

    print("Dance Data Collection Script\n" "============================\n")

    data = []

    # Get type
    line = ""
    while not (line == "m" or line == "s"):
        line = input(
            "Do you want to record multiple moves(m) or just a single move(s)? "
        )

    # Get type
    pos = ""
    while not (pos == "y" or pos == "n"):
        pos = input("Are you collecting positional data? (y/n) ")

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

    timeout = 2
    while timeout > 0:
        clr()
        print("Starting in %ds, please stand in calibration position" % timeout)
        timeout = timeout - 1
        time.sleep(1)

    intcomm = IntComm(0)

    for i in range(NUM_MOVES + 1):
        move = moves[randrange(len(moves))]
        if pos == "y":
            move = positions[i % (len(positions))] + "+" + move

        clr()

        if i == 0:
            msg = "Don't move"
        else:
            msg = move
        print("%d: %s" % (i, msg))

        data_c = 0
        while data_c < NUM_S_PER_MOVE:
            try:
                line = intcomm.get_line()
            except:
                line = ""

            if len(line) == 0 or line[0] != "#" or "#" in line[1:]:
                print("Invalid", line)
                continue
            point = str(i) + "," + line[1:]
            if pos == "y":
                point = point + "," + positions[i % (len(positions))]
            else:
                point = point + "," + move
            data_c = data_c + 1

            if i > 0:
                data.append(point)

    data = "\n".join(data) + "\n"

    line = ""
    while not (line == "y" or line == "n"):
        line = input("Do you want to keep the data? (y/n) ")

    if line == "y":
        f = open("data.csv", "a+")
        f.write(data)
        f.close()
