from ultra96 import (
    ACTIONS,
    tabulate_dance_moves,
    tabulate_positions,
    tabulate_results,
    tabulate_sync_delay,
)


def test_tabulate_dance_moves_no_error():
    dancer_moves = ["gun", "gun", "gun"]
    dancer_accuracies = [5, 2, 7]
    main_dancer_id = 0
    guest_dancer_id = 1
    dance_move, accuracy = tabulate_dance_moves(
        dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
    )
    print(dance_move, accuracy)
    assert dance_move == "gun"
    assert accuracy == 5


def test_tabulate_dance_moves_main_dancer_none_error():
    dancer_moves = [None, "gun", "sidepump"]
    dancer_accuracies = [5, 2, 7]
    main_dancer_id = 0
    guest_dancer_id = 1
    dance_move, accuracy = tabulate_dance_moves(
        dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
    )
    print(dance_move, accuracy)
    assert dance_move == "sidepump"
    assert accuracy == 7


def test_tabulate_dance_moves_main_and_guest_dancer_none_error():
    dancer_moves = [None, "gun", None]
    dancer_accuracies = [5, 2, 7]
    main_dancer_id = 0
    guest_dancer_id = 1
    dance_move, accuracy = tabulate_dance_moves(
        dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
    )
    print(dance_move, accuracy)
    assert dance_move == "gun"
    assert accuracy == 2


def test_tabulate_dance_moves_all_dancers_none_error():
    dancer_moves = [None, None, None]
    dancer_accuracies = [5, 2, 7]
    main_dancer_id = 0
    guest_dancer_id = 1
    dance_move, accuracy = tabulate_dance_moves(
        dancer_moves, dancer_accuracies, main_dancer_id, guest_dancer_id
    )
    print(dance_move, accuracy)
    assert dance_move in ACTIONS
    assert accuracy in [50.92, 50.72, 50.23]


def tabulate_sync_delay_no_error():
    dancer_start_times = [1.2, 1.3, 1.1]
    sync_delay = tabulate_sync_delay(dancer_start_times)
    print(sync_delay)
    assert sync_delay <= 0.2 and sync_delay >= 0


def tabulate_sync_delay_one_error():
    dancer_start_times = [1.2, None, 1.1]
    sync_delay = tabulate_sync_delay(dancer_start_times)
    print(sync_delay)
    assert sync_delay <= 0.1 and sync_delay >= 0


def tabulate_sync_delay_two_error():
    dancer_start_times = [1.2, None, None]
    sync_delay = tabulate_sync_delay(dancer_start_times)
    print(sync_delay)
    assert sync_delay in [0.454965591436666, 0.3437284708026666, 0.248802185056666]


def tabulate_sync_delay_three_error():
    dancer_start_times = [None, None, None]
    sync_delay = tabulate_sync_delay(dancer_start_times)
    print(sync_delay)
    assert sync_delay in [0.454965591436666, 0.3437284708026666, 0.248802185056666]


def tabulate_positions_no_errors():
    dancer_positions = [1, 1, -2]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 1, 2]


def tabulate_positions_no_errors2():
    dancer_positions = [1, -2, 1]
    original_positions = [3, 1, 2]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [2, 3, 1]


def tabulate_positions_no_errors3():
    dancer_positions = [-2, 1, 1]
    original_positions = [2, 3, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [1, 2, 3]


def tabulate_positions_no_errors4():
    dancer_positions = [2, -1, -1]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [2, 3, 1]


def tabulate_positions_no_errors5():
    dancer_positions = [-1, 2, -1]
    original_positions = [2, 3, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 1, 2]


def tabulate_positions_no_errors6():
    dancer_positions = [-1, -1, 2]
    original_positions = [3, 1, 2]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [1, 2, 3]


def tabulate_positions_no_errors7():
    dancer_positions = [0, 0, 0]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [1, 2, 3]


def tabulate_positions_no_errors8():
    dancer_positions = [0, 1, -1]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [1, 3, 2]


def tabulate_positions_no_errors9():
    dancer_positions = [1, -1, 0]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [2, 1, 3]


def tabulate_positions_no_errors10():
    dancer_positions = [2, 0, -2]
    original_positions = [1, 2, 3]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 2, 1]


def tabulate_positions_no_errors11():
    dancer_positions = [-2, 2, 0]
    original_positions = [2, 3, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [1, 3, 2]


def tabulate_positions_no_errors12():
    dancer_positions = [-1, 1, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 1, 2]


def tabulate_positions_below_threshold_error():
    dancer_positions = [1, -4, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [2, 3, 1]


def tabulate_positions_below_threshold_error2():
    dancer_positions = [1, -4, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 0
    guest_dancer_id = 1
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 2, 1]


def tabulate_positions_above_threshold_error():
    dancer_positions = [1, 5, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 0
    guest_dancer_id = 2
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 2, 1] or positions == [2, 3, 1]


def tabulate_positions_above_threshold_error2():
    dancer_positions = [1, 5, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 0
    guest_dancer_id = 1
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 2, 1]


def tabulate_positions_main_dancer1():
    dancer_positions = [1, 5, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 1
    guest_dancer_id = 0
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 1, 2]


def tabulate_positions_main_dancer2():
    dancer_positions = [1, 5, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 2
    guest_dancer_id = 1
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 2, 1]


def tabulate_positions_main_dancer3():
    dancer_positions = [1, 5, 0]
    original_positions = [3, 2, 1]
    main_dancer_id = 2
    guest_dancer_id = 0
    positions = tabulate_positions(
        dancer_positions, original_positions, main_dancer_id, guest_dancer_id
    )
    print(positions)
    assert positions == [3, 1, 2]


def tabulate_results_no_error():
    dancer_readiness = [True, True, True]
    dancer_start_times = [1.2, 1.3, 1.1]
    dancer_moves = ["gun", "gun", "hair"]
    dancer_accuracies = [40.12, 30.12, 50.2]
    dancer_positions = [1, -2, 1]
    original_positions = [3, 1, 2]
    main_dancer_id = 0
    guest_dancer_id = 2

    dance_move, sync_delay, positions, accuracy = tabulate_results(
        dancer_readiness,
        dancer_start_times,
        dancer_moves,
        dancer_accuracies,
        dancer_positions,
        original_positions,
        main_dancer_id,
        guest_dancer_id,
    )
    print(dance_move, sync_delay, positions, accuracy)
    assert sync_delay != -1
    assert dance_move == "gun"
    assert positions == [2, 3, 1]
    assert accuracy == 40.12


if __name__ == "__main__":
    test_tabulate_dance_moves_no_error()
    test_tabulate_dance_moves_main_dancer_none_error()
    test_tabulate_dance_moves_main_and_guest_dancer_none_error()
    test_tabulate_dance_moves_all_dancers_none_error()
    tabulate_sync_delay_no_error()
    tabulate_sync_delay_one_error()
    tabulate_sync_delay_two_error()
    tabulate_sync_delay_three_error()
    tabulate_positions_no_errors()
    tabulate_positions_no_errors2()
    tabulate_positions_no_errors3()
    tabulate_positions_no_errors4()
    tabulate_positions_no_errors5()
    tabulate_positions_no_errors6()
    tabulate_positions_no_errors7()
    tabulate_positions_no_errors8()
    tabulate_positions_no_errors9()
    tabulate_positions_no_errors10()
    tabulate_positions_no_errors11()
    tabulate_positions_no_errors12()
    tabulate_positions_below_threshold_error()
    tabulate_positions_below_threshold_error2()
    tabulate_positions_above_threshold_error()
    tabulate_positions_above_threshold_error2()
    tabulate_positions_main_dancer1()
    tabulate_positions_main_dancer2()
    tabulate_positions_main_dancer3()
    tabulate_results_no_error()
