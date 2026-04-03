import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from controller.estop import EStop


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        # =========================
        # Parameter declaration
        # =========================
        self.declare_parameter('control_rate_hz', 50.0)

        # E-STOP
        EStop.declare_parameters(self)

        # -------------------------
        # GAP FOLLOW
        # -------------------------
        self.declare_parameter('gf_speed', 2.0)
        self.declare_parameter('gf_max_steer', 0.4)
        self.declare_parameter('gf_max_range', 10.0)
        self.declare_parameter('gf_front_angle_deg', 70.0)
        self.declare_parameter('gf_min_gap_size', 8)
        self.declare_parameter('gf_min_valid_range', 0.05)

        self.declare_parameter('gf_corner_speed_scale', 0.7)
        self.declare_parameter('gf_min_speed', 0.5)
        self.declare_parameter('gf_close_distance', 2.0)
        self.declare_parameter('gf_very_close_distance', 1.0)
        self.declare_parameter('gf_close_speed', 1.5)
        self.declare_parameter('gf_very_close_speed', 1.0)

        # -------------------------
        # VEHICLE GEOMETRY
        # -------------------------
        self.declare_parameter('use_vehicle_geometry', True)
        self.declare_parameter('vehicle_width', 0.30)
        self.declare_parameter('vehicle_length', 0.50)
        self.declare_parameter('vehicle_wheelbase', 0.33)
        self.declare_parameter('vehicle_safety_margin', 0.05)

        # -------------------------
        # GAP / CORNER BIAS
        # -------------------------
        self.declare_parameter('gf_use_corner_bias', True)
        self.declare_parameter('gf_center_sector_deg', 12.0)
        self.declare_parameter('gf_side_sector_start_deg', 20.0)
        self.declare_parameter('gf_side_sector_end_deg', 70.0)
        self.declare_parameter('gf_turn_open_ratio', 1.25)
        self.declare_parameter('gf_turn_diff_min', 0.8)
        self.declare_parameter('gf_corner_bias_ratio', 0.78)

        # -------------------------
        # ASYMMETRIC FOV
        # -------------------------
        self.declare_parameter('gf_use_asymmetric_fov', True)
        self.declare_parameter('gf_left_fov_deg_nominal', 70.0)
        self.declare_parameter('gf_right_fov_deg_nominal', 70.0)
        self.declare_parameter('gf_left_fov_deg_turn', 85.0)
        self.declare_parameter('gf_right_fov_deg_turn', 40.0)

        # -------------------------
        # ONE-SIDE WALL FOLLOW PRIOR
        # -------------------------
        self.declare_parameter('race_follow_side', 'right')   # 'right' or 'left'
        self.declare_parameter('wf_target_dist', 0.8)
        self.declare_parameter('wf_kp', 0.9)
        self.declare_parameter('wf_max_steer', 0.30)
        self.declare_parameter('wf_theta_deg', 50.0)
        self.declare_parameter('wf_right_angle_b_deg', -90.0)
        self.declare_parameter('wf_left_angle_b_deg', 90.0)
        self.declare_parameter('wf_lookahead', 0.5)

        # wall-follow validity
        self.declare_parameter('wf_min_valid_dist', 0.15)
        self.declare_parameter('wf_max_valid_dist', 6.0)

        # -------------------------
        # HYBRID SWITCH CONDITIONS
        # -------------------------
        self.declare_parameter('hybrid_nearest_dist_threshold', 1.5)
        self.declare_parameter('hybrid_gap_width_threshold', 1.20)
        self.declare_parameter('hybrid_gap_override_steer_deg', 10.0)
        self.declare_parameter('hybrid_disable_wall_on_turn_hint', True)
        self.declare_parameter('hybrid_wall_weight', 1.0)
        self.declare_parameter('hybrid_gap_weight', 1.0)

        # -------------------------
        # STEERING SMOOTHING
        # -------------------------
        self.declare_parameter('gf_use_steer_rate_limit', True)
        self.declare_parameter('gf_max_steer_rate', 1.8)

        # -------------------------
        # VISUALIZATION
        # -------------------------
        self.declare_parameter('viz_frame', 'base_link')
        self.declare_parameter('viz_best_point_topic', '/gf/best_point_marker')
        self.declare_parameter('viz_nearest_point_topic', '/gf/nearest_point_marker')
        self.declare_parameter('viz_bubble_topic', '/gf/bubble_marker')
        self.declare_parameter('viz_gap_topic', '/gf/gap_marker')
        self.declare_parameter('viz_footprint_topic', '/gf/footprint_marker')
        self.declare_parameter('viz_fov_topic', '/gf/fov_marker')

        p = lambda name: self.get_parameter(name).value

        # =========================
        # Load params
        # =========================
        self.control_rate_hz = float(p('control_rate_hz'))

        self.gf_speed = float(p('gf_speed'))
        self.max_steer = float(p('gf_max_steer'))
        self.max_range = float(p('gf_max_range'))
        self.front_angle_deg = float(p('gf_front_angle_deg'))
        self.min_gap_size = int(p('gf_min_gap_size'))
        self.min_valid_range = float(p('gf_min_valid_range'))

        self.corner_speed_scale = float(p('gf_corner_speed_scale'))
        self.min_speed = float(p('gf_min_speed'))
        self.close_distance = float(p('gf_close_distance'))
        self.very_close_distance = float(p('gf_very_close_distance'))
        self.close_speed = float(p('gf_close_speed'))
        self.very_close_speed = float(p('gf_very_close_speed'))

        self.use_vehicle_geometry = bool(p('use_vehicle_geometry'))
        self.vehicle_width = float(p('vehicle_width'))
        self.vehicle_length = float(p('vehicle_length'))
        self.vehicle_wheelbase = float(p('vehicle_wheelbase'))
        self.vehicle_safety_margin = float(p('vehicle_safety_margin'))

        self.use_corner_bias = bool(p('gf_use_corner_bias'))
        self.center_sector_deg = float(p('gf_center_sector_deg'))
        self.side_sector_start_deg = float(p('gf_side_sector_start_deg'))
        self.side_sector_end_deg = float(p('gf_side_sector_end_deg'))
        self.turn_open_ratio = float(p('gf_turn_open_ratio'))
        self.turn_diff_min = float(p('gf_turn_diff_min'))
        self.corner_bias_ratio = float(p('gf_corner_bias_ratio'))

        self.use_asymmetric_fov = bool(p('gf_use_asymmetric_fov'))
        self.left_fov_deg_nominal = float(p('gf_left_fov_deg_nominal'))
        self.right_fov_deg_nominal = float(p('gf_right_fov_deg_nominal'))
        self.left_fov_deg_turn = float(p('gf_left_fov_deg_turn'))
        self.right_fov_deg_turn = float(p('gf_right_fov_deg_turn'))

        self.race_follow_side = str(p('race_follow_side')).lower()
        self.wf_target_dist = float(p('wf_target_dist'))
        self.wf_kp = float(p('wf_kp'))
        self.wf_max_steer = float(p('wf_max_steer'))
        self.wf_theta_deg = float(p('wf_theta_deg'))
        self.wf_right_angle_b_deg = float(p('wf_right_angle_b_deg'))
        self.wf_left_angle_b_deg = float(p('wf_left_angle_b_deg'))
        self.wf_lookahead = float(p('wf_lookahead'))
        self.wf_min_valid_dist = float(p('wf_min_valid_dist'))
        self.wf_max_valid_dist = float(p('wf_max_valid_dist'))

        self.hybrid_nearest_dist_threshold = float(p('hybrid_nearest_dist_threshold'))
        self.hybrid_gap_width_threshold = float(p('hybrid_gap_width_threshold'))
        self.hybrid_gap_override_steer_deg = float(p('hybrid_gap_override_steer_deg'))
        self.hybrid_disable_wall_on_turn_hint = bool(p('hybrid_disable_wall_on_turn_hint'))
        self.hybrid_wall_weight = float(p('hybrid_wall_weight'))
        self.hybrid_gap_weight = float(p('hybrid_gap_weight'))

        self.use_steer_rate_limit = bool(p('gf_use_steer_rate_limit'))
        self.max_steer_rate = float(p('gf_max_steer_rate'))

        self.viz_frame = str(p('viz_frame'))
        self.viz_best_point_topic = str(p('viz_best_point_topic'))
        self.viz_nearest_point_topic = str(p('viz_nearest_point_topic'))
        self.viz_bubble_topic = str(p('viz_bubble_topic'))
        self.viz_gap_topic = str(p('viz_gap_topic'))
        self.viz_footprint_topic = str(p('viz_footprint_topic'))
        self.viz_fov_topic = str(p('viz_fov_topic'))

        # =========================
        # State
        # =========================
        self.scan = None
        self.odom = None
        self.prev_steer = 0.0
        self.prev_time = None

        self.estop = EStop(self)

        # =========================
        # ROS interfaces
        # =========================
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            10
        )

        self.best_point_pub = self.create_publisher(Marker, self.viz_best_point_topic, 10)
        self.nearest_point_pub = self.create_publisher(Marker, self.viz_nearest_point_topic, 10)
        self.bubble_pub = self.create_publisher(Marker, self.viz_bubble_topic, 10)
        self.gap_pub = self.create_publisher(Marker, self.viz_gap_topic, 10)
        self.footprint_pub = self.create_publisher(Marker, self.viz_footprint_topic, 10)
        self.fov_pub = self.create_publisher(Marker, self.viz_fov_topic, 10)

        self.create_timer(1.0 / self.control_rate_hz, self._loop)

        self.get_logger().info('GapFollowNode ready (race-direction hybrid)')

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is None:
            dt = 1.0 / self.control_rate_hz
        else:
            dt = now - self.prev_time
            if dt <= 1e-4:
                dt = 1.0 / self.control_rate_hz
            elif dt > 0.2:
                dt = 0.2
        self.prev_time = now

        result = self._compute(dt)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(result['steering'])
        msg.drive.speed = float(result['speed'])

        msg = self.estop.should_stop(self.scan, self.odom, msg)
        self.drive_pub.publish(msg)

        self._publish_visualization(result)

    def _compute(self, dt):
        scan = self.scan
        ranges = np.array(scan.ranges, dtype=np.float32)

        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        n = len(ranges)

        # 1) turn hint
        turn_hint = self._detect_turn_hint(ranges, angle_min, angle_inc, n)

        # 2) FOV selection
        left_fov_deg, right_fov_deg = self._get_fov_by_turn_hint(turn_hint)

        start_idx = self._angle_to_index(-math.radians(right_fov_deg), angle_min, angle_inc, n)
        end_idx = self._angle_to_index(math.radians(left_fov_deg), angle_min, angle_inc, n)

        proc = ranges.copy()
        proc[:start_idx] = 0.0
        proc[end_idx + 1:] = 0.0
        proc[proc < self.min_valid_range] = 0.0

        # 3) nearest obstacle
        front_ranges = proc[start_idx:end_idx + 1]
        valid = np.where(front_ranges > 0.0)[0]
        if len(valid) == 0:
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc)

        local_min = valid[np.argmin(front_ranges[valid])]
        nearest_idx = start_idx + int(local_min)
        nearest_dist = float(proc[nearest_idx])

        # 4) vehicle-width-based bubble
        effective_radius = self._get_effective_bubble_radius()

        if nearest_dist > effective_radius:
            theta = math.atan2(effective_radius, nearest_dist)
        else:
            theta = math.pi / 2.0

        bubble_size = int(theta / angle_inc)
        bubble_start = max(0, nearest_idx - bubble_size)
        bubble_end = min(n - 1, nearest_idx + bubble_size)
        proc[bubble_start:bubble_end + 1] = 0.0

        # 5) gap search
        free = np.where(proc > 0.0)[0]
        if len(free) == 0:
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc, nearest_idx, nearest_dist, effective_radius)

        splits = np.split(free, np.where(np.diff(free) > 1)[0] + 1)
        gaps = [g for g in splits if len(g) >= self.min_gap_size]
        if len(gaps) == 0:
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc, nearest_idx, nearest_dist, effective_radius)

        best_gap = max(gaps, key=lambda g: len(g))
        gap_start = int(best_gap[0])
        gap_end = int(best_gap[-1])

        # 6) gap-follow steer
        best_idx = self._select_best_point(gap_start, gap_end, turn_hint)
        best_dist = float(proc[best_idx])
        best_angle = angle_min + best_idx * angle_inc
        gap_steer = float(np.clip(best_angle, -self.max_steer, self.max_steer))
        gap_width_est = self._estimate_gap_width(gap_start, gap_end, max(best_dist, 0.8), angle_inc)

        # 7) one-side wall-follow prior
        wall_valid, wall_steer, wall_debug = self._compute_one_side_wall_follow(
            ranges, angle_min, angle_inc, n
        )

        # 8) hybrid switching
        use_gap_override = False

        if nearest_dist < self.hybrid_nearest_dist_threshold:
            use_gap_override = True

        if gap_width_est < self.hybrid_gap_width_threshold:
            use_gap_override = True

        if abs(math.degrees(gap_steer)) > self.hybrid_gap_override_steer_deg:
            use_gap_override = True

        if self.hybrid_disable_wall_on_turn_hint and turn_hint != 'center':
            use_gap_override = True

        if not wall_valid:
            use_gap_override = True

        if use_gap_override:
            steering = gap_steer
            mode = 'GAP'
        else:
            steering = self.hybrid_wall_weight * wall_steer + self.hybrid_gap_weight * gap_steer
            mode = 'BLEND'

        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        if self.use_steer_rate_limit:
            steering = self._apply_steer_rate_limit(steering, dt)

        # 9) speed
        steer_ratio = abs(steering) / self.max_steer if self.max_steer > 1e-6 else 0.0
        speed = self.gf_speed * (1.0 - self.corner_speed_scale * steer_ratio)

        if best_dist < self.close_distance:
            speed = min(speed, self.close_speed)
        if best_dist < self.very_close_distance:
            speed = min(speed, self.very_close_speed)

        speed = max(speed, self.min_speed)

        self.get_logger().info(
            f'[RACE] mode={mode} | steer={math.degrees(steering):.1f} deg | '
            f'gap_steer={math.degrees(gap_steer):.1f} deg | '
            f'wall_steer={math.degrees(wall_steer):.1f} deg | '
            f'nearest={nearest_dist:.2f} | gap_width={gap_width_est:.2f} | '
            f'turn_hint={turn_hint} | '
            f'wall_valid={wall_valid} | '
            f'wall_a={wall_debug["a"]:.2f} | wall_b={wall_debug["b"]:.2f}'
        )

        return {
            'steering': steering,
            'speed': float(speed),
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'gap_start': gap_start,
            'gap_end': gap_end,
            'best_idx': best_idx,
            'best_dist': best_dist,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'angle_min': angle_min,
            'angle_inc': angle_inc,
            'effective_radius': effective_radius,
        }

    def _empty_result(self, start_idx, end_idx, angle_min, angle_inc,
                      nearest_idx=None, nearest_dist=None, effective_radius=None):
        if effective_radius is None:
            effective_radius = self._get_effective_bubble_radius()
        return {
            'steering': 0.0,
            'speed': 0.0,
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'gap_start': None,
            'gap_end': None,
            'best_idx': None,
            'best_dist': None,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'angle_min': angle_min,
            'angle_inc': angle_inc,
            'effective_radius': effective_radius,
        }

    def _get_effective_bubble_radius(self):
        if self.use_vehicle_geometry:
            return self.vehicle_width / 2.0 + self.vehicle_safety_margin
        return self.vehicle_width / 2.0 + self.vehicle_safety_margin

    def _angle_to_index(self, angle_rad, angle_min, angle_inc, n):
        idx = int((angle_rad - angle_min) / angle_inc)
        return max(0, min(n - 1, idx))

    def _get_range_at(self, ranges, angle_min, angle_inc, angle_rad):
        f_idx = (angle_rad - angle_min) / angle_inc

        if f_idx < 0 or f_idx >= len(ranges) - 1:
            return 100.0

        i0 = int(math.floor(f_idx))
        i1 = int(math.ceil(f_idx))

        r0 = ranges[i0]
        r1 = ranges[i1]

        if not np.isfinite(r0):
            r0 = 100.0
        if not np.isfinite(r1):
            r1 = 100.0

        w = f_idx - i0
        return float(r0 * (1.0 - w) + r1 * w)

    def _sector_stat(self, ranges, angle_min, angle_inc, n, angle_start_deg, angle_end_deg):
        a0 = math.radians(angle_start_deg)
        a1 = math.radians(angle_end_deg)
        i0 = self._angle_to_index(a0, angle_min, angle_inc, n)
        i1 = self._angle_to_index(a1, angle_min, angle_inc, n)
        if i0 > i1:
            i0, i1 = i1, i0
        seg = ranges[i0:i1 + 1]
        valid = seg[seg > self.min_valid_range]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    def _detect_turn_hint(self, ranges, angle_min, angle_inc, n):
        if not self.use_corner_bias:
            return 'center'

        center_half = self.center_sector_deg
        side_s = self.side_sector_start_deg
        side_e = min(self.side_sector_end_deg, self.front_angle_deg)

        center_score = self._sector_stat(ranges, angle_min, angle_inc, n, -center_half, center_half)
        left_score = self._sector_stat(ranges, angle_min, angle_inc, n, side_s, side_e)
        right_score = self._sector_stat(ranges, angle_min, angle_inc, n, -side_e, -side_s)

        if left_score > center_score * self.turn_open_ratio and (left_score - right_score) > self.turn_diff_min:
            return 'left'
        if right_score > center_score * self.turn_open_ratio and (right_score - left_score) > self.turn_diff_min:
            return 'right'
        return 'center'

    def _get_fov_by_turn_hint(self, turn_hint):
        if not self.use_asymmetric_fov:
            return self.left_fov_deg_nominal, self.right_fov_deg_nominal

        if turn_hint == 'left':
            return self.left_fov_deg_turn, self.right_fov_deg_turn
        if turn_hint == 'right':
            return self.right_fov_deg_turn, self.left_fov_deg_turn
        return self.left_fov_deg_nominal, self.right_fov_deg_nominal

    def _select_best_point(self, gap_start, gap_end, turn_hint):
        if gap_start is None or gap_end is None:
            return None
        gap_len = gap_end - gap_start
        if gap_len <= 0:
            return gap_start
        if turn_hint == 'center':
            return (gap_start + gap_end) // 2
        if turn_hint == 'left':
            idx = int(gap_start + self.corner_bias_ratio * gap_len)
            return max(gap_start, min(gap_end, idx))
        if turn_hint == 'right':
            idx = int(gap_start + (1.0 - self.corner_bias_ratio) * gap_len)
            return max(gap_start, min(gap_end, idx))
        return (gap_start + gap_end) // 2

    def _estimate_gap_width(self, gap_start, gap_end, rep_dist, angle_inc):
        angular_width = max(0.0, (gap_end - gap_start) * angle_inc)
        return float(2.0 * rep_dist * math.sin(angular_width / 2.0))

    def _compute_one_side_wall_follow(self, ranges, angle_min, angle_inc, n):
        theta = math.radians(self.wf_theta_deg)

        if self.race_follow_side == 'right':
            angle_b = math.radians(self.wf_right_angle_b_deg)
        else:
            angle_b = math.radians(self.wf_left_angle_b_deg)

        angle_a = angle_b + theta if self.race_follow_side == 'right' else angle_b - theta

        a = self._get_range_at(ranges, angle_min, angle_inc, angle_a)
        b = self._get_range_at(ranges, angle_min, angle_inc, angle_b)

        valid = (
            self.wf_min_valid_dist < a < self.wf_max_valid_dist and
            self.wf_min_valid_dist < b < self.wf_max_valid_dist
        )

        if not valid:
            return False, 0.0, {'a': a, 'b': b}

        denominator = a * math.sin(theta)
        if abs(denominator) < 1e-6:
            alpha = 0.0
        else:
            numerator = a * math.cos(theta) - b
            alpha = math.atan2(numerator, denominator)

        D_t = b * math.cos(alpha)
        D_t1 = D_t + self.wf_lookahead * math.sin(alpha)

        error = self.wf_target_dist - D_t1

        # right wall 기준이면 error > 0 -> 벽에서 멀어짐 -> 오른쪽으로 가야 함(음의 steer)
        if self.race_follow_side == 'right':
            raw_steer = -self.wf_kp * error
        else:
            raw_steer = self.wf_kp * error

        steer = float(np.clip(raw_steer, -self.wf_max_steer, self.wf_max_steer))
        return True, steer, {'a': a, 'b': b}

    def _apply_steer_rate_limit(self, target_steer, dt):
        max_delta = self.max_steer_rate * dt
        delta = target_steer - self.prev_steer
        delta = max(-max_delta, min(max_delta, delta))
        limited = self.prev_steer + delta
        self.prev_steer = limited
        return float(np.clip(limited, -self.max_steer, self.max_steer))

    def _polar_to_xy(self, idx, dist, angle_min, angle_inc):
        angle = angle_min + idx * angle_inc
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        return x, y

    def _make_marker_header(self, marker):
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = self.get_clock().now().to_msg()

    def _publish_visualization(self, result):
        self._publish_vehicle_footprint()
        self._publish_fov(result['start_idx'], result['end_idx'], result['angle_min'], result['angle_inc'])

        if result['nearest_idx'] is not None and result['nearest_dist'] is not None:
            self._publish_nearest_point(result['nearest_idx'], result['nearest_dist'],
                                        result['angle_min'], result['angle_inc'])
            self._publish_bubble(result['nearest_idx'], result['nearest_dist'],
                                 result['angle_min'], result['angle_inc'],
                                 result['effective_radius'])

        if result['gap_start'] is not None and result['gap_end'] is not None:
            self._publish_gap(result['gap_start'], result['gap_end'],
                              result['angle_min'], result['angle_inc'])

        if result['best_idx'] is not None and result['best_dist'] is not None:
            self._publish_best_point(result['best_idx'], result['best_dist'],
                                     result['angle_min'], result['angle_inc'])

    def _publish_best_point(self, idx, dist, angle_min, angle_inc):
        x, y = self._polar_to_xy(idx, dist, angle_min, angle_inc)
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_best_point'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        self.best_point_pub.publish(marker)

    def _publish_nearest_point(self, idx, dist, angle_min, angle_inc):
        x, y = self._polar_to_xy(idx, dist, angle_min, angle_inc)
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_nearest_point'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.nearest_point_pub.publish(marker)

    def _publish_bubble(self, idx, dist, angle_min, angle_inc, effective_radius):
        x, y = self._polar_to_xy(idx, dist, angle_min, angle_inc)
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_bubble'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(effective_radius * 2.0)
        marker.scale.y = float(effective_radius * 2.0)
        marker.scale.z = 0.05
        marker.color = ColorRGBA(r=0.0, g=0.3, b=1.0, a=0.35)
        self.bubble_pub.publish(marker)

    def _publish_gap(self, gap_start, gap_end, angle_min, angle_inc):
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_gap'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        rep_dist = min(3.0, self.max_range)
        for idx in range(gap_start, gap_end + 1):
            x, y = self._polar_to_xy(idx, rep_dist, angle_min, angle_inc)
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.02
            marker.points.append(p)

        self.gap_pub.publish(marker)

    def _publish_vehicle_footprint(self):
        half_w = self.vehicle_width / 2.0
        half_l = self.vehicle_length / 2.0
        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
            (half_l, half_w),
        ]
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_vehicle_footprint'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)

        for x, y in corners:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.03
            marker.points.append(p)

        self.footprint_pub.publish(marker)

    def _publish_fov(self, start_idx, end_idx, angle_min, angle_inc):
        marker = Marker()
        self._make_marker_header(marker)
        marker.ns = 'gf_fov'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)

        radius = min(3.0, self.max_range)

        origin = Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.01
        marker.points.append(origin)

        x1, y1 = self._polar_to_xy(start_idx, radius, angle_min, angle_inc)
        p1 = Point()
        p1.x = float(x1)
        p1.y = float(y1)
        p1.z = 0.01
        marker.points.append(p1)

        step = max(1, (end_idx - start_idx) // 30 + 1)
        for idx in range(start_idx, end_idx + 1, step):
            x, y = self._polar_to_xy(idx, radius, angle_min, angle_inc)
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.01
            marker.points.append(p)

        x2, y2 = self._polar_to_xy(end_idx, radius, angle_min, angle_inc)
        p2 = Point()
        p2.x = float(x2)
        p2.y = float(y2)
        p2.z = 0.01
        marker.points.append(p2)

        marker.points.append(origin)
        self.fov_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        node.drive_pub.publish(stop_msg)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()