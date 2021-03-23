import collections

import numpy as np


class PositionDetection:
    def __init__(self, verbose):
        self.position_mode = False
        self.position_direction = 0

        # constants
        self.position_threshold = 0.15
        self.verbose = verbose

    def get_mask(self, data):
        accx_mask = np.any(data[:, 3] > self.position_threshold)
        accy_mask = np.any(data[:, 4] > self.position_threshold)
        accz_mask = np.any(data[:, 5] > self.position_threshold)
        acc_right_mask = accx_mask or accy_mask or accz_mask

        accx_mask = np.any(data[:, 3] < -self.position_threshold)
        accy_mask = np.any(data[:, 4] < -self.position_threshold)
        accz_mask = np.any(data[:, 5] < -self.position_threshold)
        acc_left_mask = accx_mask or accy_mask or accz_mask

        return acc_left_mask, acc_right_mask

    def infer(self, data):
        acc_left_mask, acc_right_mask = self.get_mask(data)

        if self.position_mode:
            if self.position_direction == -1 and acc_right_mask:
                if self.verbose:
                    print("left move detected")
                self.position_mode = False
                self.position_direction = 0
                return True, "left"
            if self.position_direction == 1 and acc_left_mask:
                if self.verbose:
                    print("right move detected")
                self.position_mode = False
                self.position_direction = 0
                return True, "right"
        else:
            if acc_left_mask:
                self.position_direction = -1
                self.position_mode = True
            if acc_right_mask:
                self.position_direction = 1
                self.position_mode = True

        return False, 0


class Inference:
    def __init__(self, verbose):
        self.idle_mode_data = collections.deque([], maxlen=3)
        self.skip_count = 0

        self.position_detection = PositionDetection(verbose)

        # constants
        self.dance_threshold = 15
        self.verbose = verbose

    def append_readings(self, yaw, pitch, roll, accx, accy, accz):
        self.idle_mode_data.append([yaw, pitch, roll, accx, accy, accz])

    def get_mask(self, data):
        yaw_mask = np.any(data[:, 0] > self.dance_threshold)
        pitch_mask = np.any(data[:, 1] > self.dance_threshold)
        roll_mask = np.any(data[:, 2] > self.dance_threshold)
        angle_front_mask = yaw_mask or pitch_mask or roll_mask

        yaw_mask = np.any(data[:, 0] < -self.dance_threshold)
        pitch_mask = np.any(data[:, 1] < -self.dance_threshold)
        roll_mask = np.any(data[:, 2] < -self.dance_threshold)
        angle_back_mask = yaw_mask or pitch_mask or roll_mask

        return angle_front_mask and angle_back_mask

    def infer(self):
        if len(self.idle_mode_data) < 3 or self.skip_count > 0:
            self.skip_count = self.skip_count - 1 if self.skip_count > 0 else 0
            return None

        data = np.array(self.idle_mode_data)

        is_idle_mode = self.get_mask(data)
        if is_idle_mode:
            return None

        is_move_detected, move = self.position_detection.infer(data)
        if is_move_detected:
            self.skip_count = 10
            return move

        return None
