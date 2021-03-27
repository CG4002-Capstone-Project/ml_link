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
    # dancer_readiness,
    # dancer_start_times,
    # dancer_moves,
    # dancer_accuracies,
    # dancer_positions,
    # original_positions,
    print()
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[0])} ready state {Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[1])} start time {Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[2])} dance move {Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[3])} dance accuracy {Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[4])} dance position {Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{str(data[5])} original position {Colors.ENDC}")
    print()
