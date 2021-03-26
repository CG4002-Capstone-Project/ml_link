import os


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def clear():
    os.system("clear")


def reset_display():
    clear()
    print(f"{Colors.FAIL}{Colors.BOLD}RESETTING... DO NOT MOVE...{Colors.ENDC}")


def dance_position_display():
    clear()
    print(f"{Colors.WARNING}{Colors.BOLD}CHANGE POSITIONS{Colors.ENDC}")


def dance_move_display():
    clear()
    print(f"{Colors.OKGREEN}{Colors.BOLD}START DANCING{Colors.ENDC}")


def ready_display(count):
    clear()
    print(f"{Colors.FAIL}{Colors.BOLD}{count}...{Colors.ENDC}")


def results_display(data):
    print()
    for datum in data:
        print(f"{Colors.OKBLUE}{Colors.BOLD}{str(datum)}{Colors.ENDC}")
    print()
