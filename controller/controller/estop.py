import math
import numpy as np


class EStop:
    def __init__(self, node):
        self._logger = node.get_logger()

        p = lambda name: node.get_parameter(name).value

        self.stop_distance = float(p('estop_stop_distance'))
        self.estop_fov_deg = float(p('estop_fov_deg'))
        self.ttc_threshold = float(p('estop_ttc_threshold'))
        self.max_range = float(p('estop_max_range'))
        self.min_speed = float(p('estop_min_speed'))

        self.use_dynamic_center = bool(p('estop_use_dynamic_center'))
        self.center_gain = float(p('estop_center_gain'))
        self.max_center_shift_deg = float(p('estop_max_center_shift_deg'))

        self._was_triggered = False

    @staticmethod
    def declare_parameters(node):
        node.declare_parameter('estop_stop_distance', 0.5)
        node.declare_parameter('estop_fov_deg', 30.0)
        node.declare_parameter('estop_ttc_threshold', 0.35)
        node.declare_parameter('estop_max_range', 10.0)
        node.declare_parameter('estop_min_speed', 0.1)

        # dynamic center
        node.declare_parameter('estop_use_dynamic_center', True)
        node.declare_parameter('estop_center_gain', 0.65)
        node.declare_parameter('estop_max_center_shift_deg', 35.0)

    def should_stop(self, scan, odom, cmd):
        """
        scan, odom, cmd(AckermannDriveStamped)를 받아서
        위험하면 정지 명령으로 바꿔서 반환
        """

        if scan is None or odom is None or cmd is None:
            return cmd

        speed = float(cmd.drive.speed)

        if abs(speed) < self.min_speed:
            self._was_triggered = False
            return cmd

        ranges = np.array(scan.ranges, dtype=np.float32)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)

        # =========================
        # Dynamic FOV center
        # =========================
        center_angle = 0.0
        if self.use_dynamic_center:
            steer = float(cmd.drive.steering_angle)
            center_angle = self.center_gain * steer
            max_shift = math.radians(self.max_center_shift_deg)
            center_angle = float(np.clip(center_angle, -max_shift, max_shift))

        half_fov = math.radians(self.estop_fov_deg) / 2.0

        beam_angles = angle_min + np.arange(len(ranges)) * angle_inc
        rel_angles = beam_angles - center_angle

        # -pi ~ pi 정규화
        rel_angles = np.arctan2(np.sin(rel_angles), np.cos(rel_angles))

        mask = np.abs(rel_angles) <= half_fov
        masked_ranges = ranges[mask]
        masked_rel_angles = rel_angles[mask]

        if len(masked_ranges) == 0:
            self._was_triggered = False
            return cmd

        # 1) immediate stop distance
        positive = masked_ranges[masked_ranges > 0.0]
        if len(positive) > 0:
            min_dist = float(np.min(positive))
            if min_dist < self.stop_distance:
                stop_cmd = cmd
                stop_cmd.drive.speed = 0.0
                self._was_triggered = True
                self._logger.warn(
                    f'EStop triggered (distance) | speed={speed:.2f}, '
                    f'min_dist={min_dist:.2f}, center={math.degrees(center_angle):.1f} deg'
                )
                return stop_cmd

        # 2) TTC check
        forward_speed = abs(speed)
        valid = (masked_ranges > 0.0)

        if np.any(valid):
            rr = masked_ranges[valid]
            aa = masked_rel_angles[valid]

            closing_speed = forward_speed * np.cos(aa)
            positive_closing = closing_speed > 1e-3

            if np.any(positive_closing):
                ttc = rr[positive_closing] / closing_speed[positive_closing]
                min_ttc = float(np.min(ttc))

                if min_ttc < self.ttc_threshold:
                    stop_cmd = cmd
                    stop_cmd.drive.speed = 0.0
                    self._was_triggered = True
                    self._logger.warn(
                        f'EStop triggered (ttc) | speed={speed:.2f}, '
                        f'min_ttc={min_ttc:.2f}, center={math.degrees(center_angle):.1f} deg'
                    )
                    return stop_cmd

        self._was_triggered = False
        return cmd