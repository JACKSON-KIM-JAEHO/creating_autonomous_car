import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class GapFollowNode(Node):
    def __init__(self):
        super().__init__('gap_follow')

        self.declare_parameter('control_rate_hz', 50.0)

        self.declare_parameter('gf_speed', 1.0)
        self.declare_parameter('gf_min_speed', 0.5)
        self.declare_parameter('gf_max_steer', 0.6)
        self.declare_parameter('gf_max_range', 8.0)
        self.declare_parameter('gf_min_valid_range', 0.05)
        self.declare_parameter('gf_gap_fov_deg', 120.0)
        self.declare_parameter('gf_nearest_fov_deg', 80.0)
        self.declare_parameter('gf_distance_cap', 6.0)
        self.declare_parameter('gf_bubble_radius', 0.35)
        self.declare_parameter('gf_min_gap_beams', 8)
        self.declare_parameter('gf_gap_open_threshold', 2.2)
        self.declare_parameter('gf_gap_open_threshold_side_scale', 0.75)

        self.declare_parameter('gf_gap_heading_weight', 0.35)
        self.declare_parameter('gf_gap_depth_weight', 3.0)
        self.declare_parameter('gf_gap_width_weight', 8.0)
        self.declare_parameter('gf_forward_gap_bonus', 3.0)
        self.declare_parameter('gf_gap_keep_bonus', 18.0)
        self.declare_parameter('gf_gap_switch_margin', 12.0)

        self.declare_parameter('gf_target_range_weight_power', 1.0)
        self.declare_parameter('gf_target_candidate_count', 9)
        self.declare_parameter('gf_target_wall_margin_deg', 8.0)
        self.declare_parameter('gf_target_clearance_weight', 35.0)
        self.declare_parameter('gf_target_range_weight', 2.0)
        self.declare_parameter('gf_target_angle_weight', 25.0)
        self.declare_parameter('gf_inside_corner_clearance', 1.1)
        self.declare_parameter('gf_inside_corner_penalty', 45.0)
        self.declare_parameter('gf_target_point_scale', 1.0)

        self.declare_parameter('gf_obstacle_trigger_dist', 3.5)
        self.declare_parameter('gf_obstacle_max_center_angle_deg', 28.0)
        self.declare_parameter('gf_obstacle_cluster_tolerance', 0.30)
        self.declare_parameter('gf_obstacle_cluster_min_beams', 3)
        self.declare_parameter('gf_obstacle_avoidance_bonus', 45.0)
        self.declare_parameter('gf_obstacle_side_clearance_weight', 16.0)
        self.declare_parameter('gf_obstacle_center_bias_weight', 28.0)

        self.declare_parameter('gf_use_footprint_clearance', True)
        self.declare_parameter('gf_footprint_use_arc', True)
        self.declare_parameter('gf_footprint_safety_margin', 0.12)
        self.declare_parameter('gf_footprint_check_dist', 2.2)
        self.declare_parameter('gf_footprint_creep_speed', 0.22)
        self.declare_parameter('gf_footprint_arc_samples', 8)
        self.declare_parameter('gf_footprint_collision_speed_scale', 0.55)

        self.declare_parameter('gf_safe_distance', 3.0)
        self.declare_parameter('gf_steer_speed_gain', 0.80)
        self.declare_parameter('gf_use_steer_smoothing', True)
        self.declare_parameter('gf_steer_smoothing_alpha', 0.50)
        self.declare_parameter('gf_use_speed_smoothing', True)
        self.declare_parameter('gf_speed_smoothing_alpha', 0.65)

        self.declare_parameter('viz_frame', 'base_link')
        self.declare_parameter('viz_rate_hz', 10.0)
        self.declare_parameter('viz_gap_fov_topic', '/gf/gap_fov_marker')
        self.declare_parameter('viz_nearest_point_topic', '/gf/nearest_point_marker')
        self.declare_parameter('viz_bubble_topic', '/gf/bubble_marker')
        self.declare_parameter('viz_gap_topic', '/gf/gap_marker')
        self.declare_parameter('viz_all_gaps_topic', '/gf/all_gaps_markers')
        self.declare_parameter('viz_gap_center_topic', '/gf/gap_center_marker')
        self.declare_parameter('viz_target_topic', '/gf/target_marker')
        self.declare_parameter('viz_footprint_topic', '/gf/footprint_marker')
        self.declare_parameter('viz_fov_line_width', 0.05)
        self.declare_parameter('viz_gap_endpoint_size', 0.18)
        self.declare_parameter('viz_all_gap_line_width', 0.04)

        self.declare_parameter('vehicle_width', 0.30)
        self.declare_parameter('vehicle_length', 0.50)
        self.declare_parameter('vehicle_wheelbase', 0.33)

        p = lambda name: self.get_parameter(name).value

        self.control_rate_hz = float(p('control_rate_hz'))
        self.nominal_speed = float(p('gf_speed'))
        self.min_speed = float(p('gf_min_speed'))
        self.max_steer = float(p('gf_max_steer'))
        self.max_range = float(p('gf_max_range'))
        self.min_valid_range = float(p('gf_min_valid_range'))
        self.gap_fov_deg = float(p('gf_gap_fov_deg'))
        self.nearest_fov_deg = float(p('gf_nearest_fov_deg'))
        self.distance_cap = float(p('gf_distance_cap'))
        self.bubble_radius = float(p('gf_bubble_radius'))
        self.min_gap_beams = int(p('gf_min_gap_beams'))
        self.gap_open_threshold = float(p('gf_gap_open_threshold'))
        self.gap_open_threshold_side_scale = float(p('gf_gap_open_threshold_side_scale'))

        self.gap_heading_weight = float(p('gf_gap_heading_weight'))
        self.gap_depth_weight = float(p('gf_gap_depth_weight'))
        self.gap_width_weight = float(p('gf_gap_width_weight'))
        self.forward_gap_bonus = float(p('gf_forward_gap_bonus'))
        self.gap_keep_bonus = float(p('gf_gap_keep_bonus'))
        self.gap_switch_margin = float(p('gf_gap_switch_margin'))

        self.target_range_weight_power = float(p('gf_target_range_weight_power'))
        self.target_candidate_count = int(p('gf_target_candidate_count'))
        self.target_wall_margin = math.radians(float(p('gf_target_wall_margin_deg')))
        self.target_clearance_weight = float(p('gf_target_clearance_weight'))
        self.target_range_weight = float(p('gf_target_range_weight'))
        self.target_angle_weight = float(p('gf_target_angle_weight'))
        self.inside_corner_clearance = float(p('gf_inside_corner_clearance'))
        self.inside_corner_penalty = float(p('gf_inside_corner_penalty'))
        self.target_point_scale = float(p('gf_target_point_scale'))

        self.obstacle_trigger_dist = float(p('gf_obstacle_trigger_dist'))
        self.obstacle_max_center_angle_deg = float(p('gf_obstacle_max_center_angle_deg'))
        self.obstacle_cluster_tolerance = float(p('gf_obstacle_cluster_tolerance'))
        self.obstacle_cluster_min_beams = int(p('gf_obstacle_cluster_min_beams'))
        self.obstacle_avoidance_bonus = float(p('gf_obstacle_avoidance_bonus'))
        self.obstacle_side_clearance_weight = float(p('gf_obstacle_side_clearance_weight'))
        self.obstacle_center_bias_weight = float(p('gf_obstacle_center_bias_weight'))

        self.use_footprint_clearance = bool(p('gf_use_footprint_clearance'))
        self.footprint_use_arc = bool(p('gf_footprint_use_arc'))
        self.footprint_safety_margin = float(p('gf_footprint_safety_margin'))
        self.footprint_check_dist = float(p('gf_footprint_check_dist'))
        self.footprint_creep_speed = float(p('gf_footprint_creep_speed'))
        self.footprint_arc_samples = int(p('gf_footprint_arc_samples'))
        self.footprint_collision_speed_scale = float(p('gf_footprint_collision_speed_scale'))

        self.safe_distance = float(p('gf_safe_distance'))
        self.steer_speed_gain = float(p('gf_steer_speed_gain'))
        self.use_steer_smoothing = bool(p('gf_use_steer_smoothing'))
        self.steer_smoothing_alpha = float(p('gf_steer_smoothing_alpha'))
        self.use_speed_smoothing = bool(p('gf_use_speed_smoothing'))
        self.speed_smoothing_alpha = float(p('gf_speed_smoothing_alpha'))

        self.viz_frame = str(p('viz_frame'))
        self.viz_rate_hz = float(p('viz_rate_hz'))
        self.viz_fov_line_width = float(p('viz_fov_line_width'))
        self.viz_gap_endpoint_size = float(p('viz_gap_endpoint_size'))
        self.viz_all_gap_line_width = float(p('viz_all_gap_line_width'))
        self.vehicle_width = float(p('vehicle_width'))
        self.vehicle_length = float(p('vehicle_length'))
        self.vehicle_wheelbase = float(p('vehicle_wheelbase'))

        self.scan = None
        self.odom = None
        self.prev_steering = 0.0
        self.prev_speed = 0.0
        self.prev_gap_center_angle = None
        self.prev_gap_side = 0
        self.debug_count = 0
        self.last_viz_time_ns = 0

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            10,
        )
        self.gap_fov_pub = self.create_publisher(Marker, str(p('viz_gap_fov_topic')), 10)
        self.nearest_pub = self.create_publisher(Marker, str(p('viz_nearest_point_topic')), 10)
        self.bubble_pub = self.create_publisher(Marker, str(p('viz_bubble_topic')), 10)
        self.gap_pub = self.create_publisher(Marker, str(p('viz_gap_topic')), 10)
        self.all_gaps_pub = self.create_publisher(MarkerArray, str(p('viz_all_gaps_topic')), 10)
        self.gap_center_pub = self.create_publisher(Marker, str(p('viz_gap_center_topic')), 10)
        self.target_pub = self.create_publisher(Marker, str(p('viz_target_topic')), 10)
        self.footprint_pub = self.create_publisher(Marker, str(p('viz_footprint_topic')), 10)

        self.create_timer(1.0 / self.control_rate_hz, self._loop)
        self.get_logger().info('GapFollowNode ready (stable simple FTG)')

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _loop(self):
        if self.scan is None:
            return

        result = self._compute()

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = float(result['steering'])
        drive_msg.drive.speed = float(result['speed'])
        self.drive_pub.publish(drive_msg)

        if self._should_publish_visualization():
            self._publish_visualization(result)

    def _should_publish_visualization(self):
        if self.viz_rate_hz <= 0.0:
            return False
        now_ns = self.get_clock().now().nanoseconds
        period_ns = int(1e9 / self.viz_rate_hz)
        if now_ns - self.last_viz_time_ns < period_ns:
            return False
        self.last_viz_time_ns = now_ns
        return True

    def _compute(self):
        ranges = np.array(self.scan.ranges, dtype=np.float32)
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)
        ranges = np.clip(ranges, 0.0, self.distance_cap)

        angles = self.scan.angle_min + np.arange(len(ranges), dtype=np.float32) * self.scan.angle_increment
        gap_mask = np.abs(angles) <= math.radians(self.gap_fov_deg) * 0.5
        nearest_mask = np.abs(angles) <= math.radians(self.nearest_fov_deg) * 0.5

        gap_ranges = np.where(gap_mask, ranges, 0.0)
        nearest_ranges = np.where(nearest_mask, ranges, 0.0)
        gap_ranges[gap_ranges < self.min_valid_range] = 0.0
        nearest_ranges[nearest_ranges < self.min_valid_range] = 0.0

        obstacle_clusters = self._select_obstacle_clusters(nearest_ranges, angles)
        nearest_idx, nearest_dist = self._nearest_from_clusters(obstacle_clusters)
        obstacle_hint = self._obstacle_avoidance_hint(obstacle_clusters, angles, ranges)
        bubble_mask = self._apply_bubbles(gap_ranges, angles, gap_mask, obstacle_clusters)
        threshold_blocked = self._apply_open_space_threshold(gap_ranges, angles)

        gaps = self._find_gaps(gap_ranges)
        gap_infos = self._score_gaps(gaps, angles, gap_ranges)
        route = self._select_gap_and_target(gap_infos, angles, ranges, gap_ranges, obstacle_hint)
        selected_gap = route['gap'] if route is not None else None
        if selected_gap is None:
            return self._fallback_result(
                angles,
                gap_mask,
                nearest_idx,
                nearest_dist,
                obstacle_clusters,
                bubble_mask,
                gap_ranges,
            )

        target_angle = route['target_angle']
        target_idx = route['target_idx']
        target_range = route['target_range']
        footprint_clearance = route['footprint_clearance']
        footprint_collision = route['footprint_collision']
        driveable_gap_count = route['driveable_gap_count']
        route_score = route['route_score']

        gap_center_angle = selected_gap['center_angle']
        self.prev_gap_center_angle = gap_center_angle
        self.prev_gap_side = self._angle_side(gap_center_angle)

        raw_steering = float(np.clip(target_angle * self.target_point_scale, -self.max_steer, self.max_steer))
        steering = raw_steering
        if self.use_steer_smoothing:
            alpha = float(np.clip(self.steer_smoothing_alpha, 0.0, 1.0))
            steering = alpha * self.prev_steering + (1.0 - alpha) * raw_steering
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        self.prev_steering = steering

        steer_ratio = abs(steering) / self.max_steer if self.max_steer > 1e-6 else 0.0
        clearance_dist = nearest_dist if nearest_dist is not None else self.safe_distance
        clearance_ratio = min(clearance_dist / self.safe_distance, 1.0)
        raw_speed = self.nominal_speed * (1.0 - self.steer_speed_gain * steer_ratio)
        raw_speed *= 0.7 + 0.3 * clearance_ratio
        if footprint_collision:
            collision_scale = float(np.clip(self.footprint_collision_speed_scale, 0.1, 1.0))
            clearance_scale = float(np.clip(footprint_clearance / max(self.footprint_check_dist, 1e-3), 0.12, 1.0))
            raw_speed *= collision_scale * clearance_scale
            raw_speed = min(raw_speed, self.footprint_creep_speed)
        min_allowed = min(self.min_speed, self.footprint_creep_speed) if footprint_collision else self.min_speed
        raw_speed = float(np.clip(raw_speed, min_allowed, self.nominal_speed))

        speed = raw_speed
        if self.use_speed_smoothing:
            alpha = float(np.clip(self.speed_smoothing_alpha, 0.0, 1.0))
            speed = alpha * self.prev_speed + (1.0 - alpha) * raw_speed
            min_allowed = min(self.min_speed, self.footprint_creep_speed) if footprint_collision else self.min_speed
            speed = float(np.clip(speed, min_allowed, self.nominal_speed))
        self.prev_speed = speed

        self.debug_count += 1
        if self.debug_count % 25 == 0:
            nearest_text = 'none' if nearest_idx is None or nearest_dist is None else f'{nearest_idx}/{nearest_dist:.2f}'
            self.get_logger().info(
                f'[GAP] nearest={nearest_text} obstacles={len(obstacle_clusters)} '
                f'hint={obstacle_hint["side"]}/{obstacle_hint["confidence"]:.2f} '
                f'gaps={len(gap_infos)}/{driveable_gap_count}driveable open_threshold={"yes" if threshold_blocked else "no"} '
                f'gap={selected_gap["start"]}-{selected_gap["end"]} score={selected_gap["score"]:.1f} '
                f'route={route_score:.1f} '
                f'center={math.degrees(gap_center_angle):.1f}deg '
                f'target={target_idx}/{math.degrees(target_angle):.1f}deg '
                f'footprint={footprint_clearance:.2f}m/{"hit" if footprint_collision else "clear"} '
                f'steer={steering:.3f} speed={speed:.3f}'
            )

        return {
            'steering': steering,
            'speed': speed,
            'angles': angles,
            'proc_ranges': gap_ranges,
            'gap_mask': gap_mask,
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'obstacle_clusters': obstacle_clusters,
            'obstacle_hint': obstacle_hint,
            'bubble_mask': bubble_mask,
            'gaps': gap_infos,
            'selected_gap': selected_gap,
            'gap_start': selected_gap['start'],
            'gap_end': selected_gap['end'],
            'target_idx': target_idx,
            'target_range': target_range,
            'target_angle': target_angle,
            'footprint_clearance': footprint_clearance,
            'footprint_collision': footprint_collision,
            'driveable_gap_count': driveable_gap_count,
            'route_score': route_score,
            'gap_center_angle': gap_center_angle,
            'used_fallback': False,
        }

    def _fallback_result(self, angles, gap_mask, nearest_idx, nearest_dist, obstacle_clusters, bubble_mask, gap_ranges):
        self.prev_gap_center_angle = None
        self.prev_gap_side = 0

        target_idx = int(np.argmin(np.abs(angles))) if len(angles) > 0 else 0
        target_angle = float(angles[target_idx]) if len(angles) > 0 else 0.0
        steering = float(np.clip(self.prev_steering, -self.max_steer, self.max_steer))
        speed = float(np.clip(self.footprint_creep_speed, 0.0, self.nominal_speed))
        self.prev_steering = steering
        self.prev_speed = speed
        self.debug_count += 1
        if self.debug_count % 25 == 0:
            nearest_text = 'none' if nearest_idx is None or nearest_dist is None else f'{nearest_idx}/{nearest_dist:.2f}'
            self.get_logger().info(f'[GAP] nearest={nearest_text} fallback=yes gap=none steer={steering:.3f} speed={speed:.3f}')

        return {
            'steering': steering,
            'speed': speed,
            'angles': angles,
            'proc_ranges': gap_ranges,
            'gap_mask': gap_mask,
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'obstacle_clusters': obstacle_clusters,
            'obstacle_hint': {'side': 0, 'confidence': 0.0, 'left_open': 0.0, 'right_open': 0.0},
            'bubble_mask': bubble_mask,
            'gaps': [],
            'selected_gap': None,
            'gap_start': None,
            'gap_end': None,
            'target_idx': target_idx,
            'target_range': 0.0,
            'target_angle': target_angle,
            'footprint_clearance': 0.0,
            'footprint_collision': True,
            'driveable_gap_count': 0,
            'route_score': 0.0,
            'gap_center_angle': target_angle,
            'used_fallback': True,
        }

    def _apply_bubbles(self, gap_ranges, angles, gap_mask, obstacle_clusters):
        bubble_mask = np.zeros_like(gap_mask, dtype=bool)
        for cluster in obstacle_clusters:
            center_angle = float(angles[cluster['center_idx']])
            half_angle = self._bubble_half_angle(cluster['dist'])
            cluster_mask = np.abs(angles - center_angle) <= half_angle
            bubble_mask |= cluster_mask & gap_mask
        gap_ranges[bubble_mask] = 0.0
        return bubble_mask

    def _bubble_half_angle(self, distance):
        distance = max(distance, self.min_valid_range)
        ratio = np.clip(self.bubble_radius / distance, -1.0, 1.0)
        return float(math.asin(ratio))

    def _select_obstacle_clusters(self, nearest_ranges, angles):
        candidate_mask = (nearest_ranges > 0.0) & (nearest_ranges <= self.obstacle_trigger_dist)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return []

        clusters = []
        cluster = []
        prev_idx = None
        prev_range = None
        tolerance = max(self.obstacle_cluster_tolerance, 0.0)

        def flush_cluster():
            if len(cluster) < self.obstacle_cluster_min_beams:
                return
            cluster_indices = np.array(cluster, dtype=int)
            center_idx = int(cluster_indices[len(cluster_indices) // 2])
            center_angle_deg = abs(math.degrees(float(angles[center_idx])))
            if center_angle_deg > self.obstacle_max_center_angle_deg:
                return
            cluster_ranges = nearest_ranges[cluster_indices]
            clusters.append({
                'start': int(cluster_indices[0]),
                'end': int(cluster_indices[-1]),
                'center_idx': center_idx,
                'dist': float(np.mean(cluster_ranges)),
                'min_dist': float(np.min(cluster_ranges)),
            })

        for idx in candidate_indices:
            idx = int(idx)
            value = float(nearest_ranges[idx])
            starts_new = (
                prev_idx is None
                or idx != prev_idx + 1
                or (prev_range is not None and abs(value - prev_range) > tolerance)
            )
            if starts_new:
                flush_cluster()
                cluster = [idx]
            else:
                cluster.append(idx)
            prev_idx = idx
            prev_range = value

        flush_cluster()
        return clusters

    def _nearest_from_clusters(self, clusters):
        if not clusters:
            return None, None
        nearest = min(clusters, key=lambda cluster: cluster['min_dist'])
        return int(nearest['center_idx']), float(nearest['dist'])

    def _obstacle_avoidance_hint(self, clusters, angles, ranges):
        empty_hint = {'side': 0, 'confidence': 0.0, 'left_open': 0.0, 'right_open': 0.0}
        if not clusters:
            return empty_hint

        obstacle = min(clusters, key=lambda cluster: cluster['min_dist'])
        obstacle_dist = float(obstacle['min_dist'])
        if obstacle_dist > self.obstacle_trigger_dist:
            return empty_hint

        center_angle = float(angles[obstacle['center_idx']])
        max_center = math.radians(self.obstacle_max_center_angle_deg)
        if abs(center_angle) > max_center:
            return empty_hint

        left_open = self._sector_open_distance(angles, ranges, math.radians(8.0), math.radians(58.0))
        right_open = self._sector_open_distance(angles, ranges, math.radians(-58.0), math.radians(-8.0))
        side_score = self.obstacle_side_clearance_weight * (left_open - right_open)
        side_score -= self.obstacle_center_bias_weight * math.degrees(center_angle) / max(self.obstacle_max_center_angle_deg, 1e-3)

        if side_score > 0.05:
            side = 1
        elif side_score < -0.05:
            side = -1
        else:
            side = 0
        if side == 0:
            side = -self._angle_side(center_angle)
        if side == 0:
            side = 1 if left_open >= right_open else -1

        openness_conf = min(abs(left_open - right_open) / max(self.distance_cap, 1e-3), 1.0)
        distance_conf = float(np.clip((self.obstacle_trigger_dist - obstacle_dist) / max(self.obstacle_trigger_dist, 1e-3), 0.0, 1.0))
        center_conf = float(np.clip(1.0 - abs(center_angle) / max(max_center, 1e-3), 0.0, 1.0))
        confidence = float(np.clip(0.25 + 0.45 * distance_conf + 0.20 * center_conf + 0.30 * openness_conf, 0.0, 1.0))

        return {
            'side': side,
            'confidence': confidence,
            'left_open': left_open,
            'right_open': right_open,
        }

    def _sector_open_distance(self, angles, ranges, start_angle, end_angle):
        low = min(start_angle, end_angle)
        high = max(start_angle, end_angle)
        mask = (angles >= low) & (angles <= high) & (ranges > self.min_valid_range)
        if not np.any(mask):
            return 0.0
        return float(np.percentile(ranges[mask], 70))

    def _apply_open_space_threshold(self, gap_ranges, angles):
        side_scale = float(np.clip(self.gap_open_threshold_side_scale, 0.1, 1.0))
        angle_scale = side_scale + (1.0 - side_scale) * np.abs(np.cos(angles))
        min_open_ranges = self.gap_open_threshold * angle_scale
        blocked_mask = (gap_ranges > 0.0) & (gap_ranges < min_open_ranges)
        gap_ranges[blocked_mask] = 0.0
        return bool(np.any(blocked_mask))

    def _find_gaps(self, proc_ranges):
        valid = proc_ranges > 0.0
        gaps = []
        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                if idx - start >= self.min_gap_beams:
                    gaps.append((start, idx - 1))
                start = None

        if start is not None and len(valid) - start >= self.min_gap_beams:
            gaps.append((start, len(valid) - 1))
        return gaps

    def _score_gaps(self, gaps, angles, gap_ranges):
        forward_idx = int(np.argmin(np.abs(angles)))
        gap_infos = []
        for start, end in gaps:
            gap_indices = np.arange(start, end + 1, dtype=int)
            ranges = gap_ranges[gap_indices]
            valid = ranges > 0.0
            if not np.any(valid):
                continue
            ranges = ranges[valid]
            center_angle = self._compute_gap_center_angle(start, end, angles, gap_ranges)
            depth = float(np.percentile(ranges, 70))
            angular_width = abs(float(angles[end] - angles[start]))
            physical_width = float(2.0 * depth * math.sin(max(angular_width, 0.0) * 0.5))
            center_angle_deg = abs(math.degrees(center_angle))

            score = 0.0
            score += self.gap_width_weight * physical_width
            score += self.gap_depth_weight * depth
            score -= self.gap_heading_weight * center_angle_deg
            if start <= forward_idx <= end:
                score += self.forward_gap_bonus
            if self.prev_gap_center_angle is not None:
                low = min(float(angles[start]), float(angles[end]))
                high = max(float(angles[start]), float(angles[end]))
                if low <= self.prev_gap_center_angle <= high:
                    score += self.gap_keep_bonus

            gap_infos.append({
                'start': int(start),
                'end': int(end),
                'center_angle': float(center_angle),
                'depth': depth,
                'width': physical_width,
                'score': float(score),
                'side': self._angle_side(center_angle),
            })
        return gap_infos

    def _select_gap(self, gaps, angles):
        if not gaps:
            return None
        best_gap = max(gaps, key=lambda gap: gap['score'])
        sticky_gap = self._find_sticky_gap(gaps, angles)
        if sticky_gap is not None and sticky_gap['score'] + self.gap_switch_margin >= best_gap['score']:
            return sticky_gap
        return best_gap

    def _select_gap_and_target(self, gaps, angles, ranges, gap_ranges, obstacle_hint):
        if not gaps:
            return None

        preferred_gap = self._select_gap(gaps, angles)
        ordered_gaps = sorted(gaps, key=lambda gap: gap['score'], reverse=True)
        if preferred_gap is not None:
            ordered_gaps = [
                preferred_gap,
                *[
                    gap for gap in ordered_gaps
                    if gap['start'] != preferred_gap['start'] or gap['end'] != preferred_gap['end']
                ],
            ]

        best_route = None
        routes = []
        for gap in ordered_gaps:
            target_angle, target_idx, target_range, clearance, collision, driveable = self._select_target(
                gap,
                angles,
                ranges,
                gap_ranges,
                obstacle_hint,
            )
            route_score = self._route_score(
                gap,
                target_angle,
                clearance,
                collision,
                preferred_gap,
                obstacle_hint,
            )
            route = {
                'gap': gap,
                'target_angle': target_angle,
                'target_idx': target_idx,
                'target_range': target_range,
                'footprint_clearance': clearance,
                'footprint_collision': collision,
                'driveable': driveable,
                'route_score': route_score,
            }
            gap['driveable'] = driveable
            gap['target_clearance'] = clearance
            gap['target_collision'] = collision
            gap['route_score'] = route_score
            routes.append(route)

        driveable_routes = [route for route in routes if route['driveable']]
        candidate_routes = driveable_routes if driveable_routes else routes
        for route in candidate_routes:
            if best_route is None or route['route_score'] > best_route['route_score']:
                best_route = route

        if best_route is not None:
            best_route['driveable_gap_count'] = len(driveable_routes)
        return best_route

    def _route_score(self, gap, target_angle, clearance, collision, preferred_gap, obstacle_hint):
        score = float(gap['score'])
        if preferred_gap is not None and gap['start'] == preferred_gap['start'] and gap['end'] == preferred_gap['end']:
            score += 6.0

        if not collision:
            score += 80.0
        else:
            clearance_ratio = float(np.clip(clearance / max(self.footprint_check_dist, 1e-3), 0.0, 1.0))
            score += 35.0 * clearance_ratio - 35.0

        score += self._obstacle_side_bonus(target_angle, obstacle_hint)

        score -= 2.0 * abs(math.degrees(target_angle - self.prev_steering))
        return float(score)

    def _obstacle_side_bonus(self, target_angle, obstacle_hint):
        hint_side = int(obstacle_hint.get('side', 0))
        confidence = float(obstacle_hint.get('confidence', 0.0))
        if hint_side == 0 or confidence <= 0.0:
            return 0.0
        target_side = self._angle_side(target_angle)
        if target_side == 0:
            return 0.0
        if target_side == hint_side:
            return self.obstacle_avoidance_bonus * confidence
        return -self.obstacle_avoidance_bonus * confidence

    def _find_sticky_gap(self, gaps, angles):
        if self.prev_gap_center_angle is None:
            return None

        candidates = []
        for gap in gaps:
            low = min(float(angles[gap['start']]), float(angles[gap['end']]))
            high = max(float(angles[gap['start']]), float(angles[gap['end']]))
            contains_previous = low <= self.prev_gap_center_angle <= high
            same_side = gap['side'] != 0 and gap['side'] == self.prev_gap_side
            if contains_previous or same_side:
                candidates.append(gap)

        if not candidates:
            return None
        return max(candidates, key=lambda gap: gap['score'])

    def _compute_gap_center_angle(self, gap_start, gap_end, angles, gap_ranges):
        gap_indices = np.arange(gap_start, gap_end + 1, dtype=int)
        valid = gap_ranges[gap_indices] > 0.0
        if not np.any(valid):
            return float(0.5 * (angles[gap_start] + angles[gap_end]))
        valid_indices = gap_indices[valid]
        return float(0.5 * (angles[valid_indices[0]] + angles[valid_indices[-1]]))

    def _select_target(self, gap, angles, ranges, gap_ranges, obstacle_hint):
        gap_start = gap['start']
        gap_end = gap['end']
        gap_indices = np.arange(gap_start, gap_end + 1, dtype=int)
        valid = gap_ranges[gap_indices] > 0.0
        candidate_indices = gap_indices[valid]
        if candidate_indices.size == 0:
            fallback_idx = int((gap_start + gap_end) // 2)
            angle = float(angles[fallback_idx])
            clearance, collision = self._path_clearance(angle, angles, ranges)
            return angle, fallback_idx, 0.0, clearance, collision, not collision

        margin = self._target_margin_for_gap(gap, angles)
        low = min(float(angles[candidate_indices[0]]), float(angles[candidate_indices[-1]])) + margin
        high = max(float(angles[candidate_indices[0]]), float(angles[candidate_indices[-1]])) - margin
        if low < high:
            candidate_indices = candidate_indices[
                (angles[candidate_indices] >= low) & (angles[candidate_indices] <= high)
            ]
        if candidate_indices.size == 0:
            candidate_indices = gap_indices[valid]

        desired_angle = self._weighted_target_angle(candidate_indices, angles, gap_ranges)
        desired_pos = int(np.argmin(np.abs(angles[candidate_indices] - desired_angle)))

        sample_positions = set()
        sample_positions.add(desired_pos)
        count = max(self.target_candidate_count, 3)
        if candidate_indices.size > count:
            sample_positions.update(np.linspace(0, candidate_indices.size - 1, count, dtype=int).tolist())
        else:
            sample_positions.update(range(candidate_indices.size))

        best_clear = None
        best_blocked = None
        clear_count = 0
        for pos in sorted(sample_positions):
            idx = int(candidate_indices[pos])
            angle = float(angles[idx])
            clearance, collision = self._path_clearance(angle, angles, ranges)
            angle_error = abs(angle - desired_angle)
            steer_change = abs(angle - self.prev_steering)
            inside_clearance = self._inside_corner_clearance(angle, angles, ranges)
            inside_deficit = max(self.inside_corner_clearance - inside_clearance, 0.0)
            inside_penalty = (
                self.inside_corner_penalty
                * inside_deficit
                * min(abs(math.degrees(angle)) / 30.0, 1.5)
            )
            score = self._target_candidate_score(
                angle,
                clearance,
                collision,
                float(gap_ranges[idx]),
                angle_error,
                steer_change,
                inside_penalty,
                obstacle_hint,
            )
            candidate = (score, angle, idx, clearance, collision)
            if collision:
                if best_blocked is None or candidate[0] > best_blocked[0]:
                    best_blocked = candidate
            else:
                clear_count += 1
                if best_clear is None or candidate[0] > best_clear[0]:
                    best_clear = candidate

        selected = best_clear if best_clear is not None else best_blocked
        _, angle, idx, clearance, collision = selected
        target_range = float(max(gap_ranges[idx], self.min_valid_range))
        return angle, idx, target_range, clearance, collision, clear_count > 0

    def _target_candidate_score(
        self,
        angle,
        clearance,
        collision,
        target_range,
        angle_error,
        steer_change,
        inside_penalty,
        obstacle_hint,
    ):
        score = 0.0
        if not collision:
            score += 120.0
        else:
            score -= 70.0

        clearance_ratio = float(np.clip(clearance / max(self.footprint_check_dist, 1e-3), 0.0, 1.0))
        range_ratio = float(np.clip(target_range / max(self.distance_cap, 1e-3), 0.0, 1.0))
        score += self.target_clearance_weight * clearance_ratio
        score += self.target_range_weight * range_ratio
        score -= self.target_angle_weight * abs(math.degrees(angle_error)) / 30.0
        score -= 10.0 * abs(math.degrees(steer_change)) / 30.0
        score -= inside_penalty
        score += self._obstacle_side_bonus(angle, obstacle_hint)
        return float(score)

    def _inside_corner_clearance(self, target_angle, angles, ranges):
        if abs(target_angle) < math.radians(5.0):
            return self.distance_cap

        side = 1.0 if target_angle > 0.0 else -1.0
        inner_start = max(abs(target_angle), math.radians(12.0))
        inner_end = min(inner_start + math.radians(45.0), math.radians(self.gap_fov_deg) * 0.5)
        side_angles = side * angles
        mask = (
            (side_angles >= inner_start)
            & (side_angles <= inner_end)
            & (ranges > self.min_valid_range)
        )
        if not np.any(mask):
            return self.distance_cap
        return float(np.percentile(ranges[mask], 20))

    def _target_margin_for_gap(self, gap, angles):
        depth = max(gap['depth'], self.min_valid_range)
        half_width = self.vehicle_width * 0.5 + max(self.footprint_safety_margin, 0.0)
        vehicle_margin = math.asin(float(np.clip(half_width / depth, 0.0, 0.95)))
        return max(self.target_wall_margin, vehicle_margin)

    def _weighted_target_angle(self, candidate_indices, angles, gap_ranges):
        candidate_ranges = gap_ranges[candidate_indices]
        candidate_angles = angles[candidate_indices]
        weights = np.power(np.maximum(candidate_ranges, self.min_valid_range), self.target_range_weight_power)
        x = candidate_ranges * np.cos(candidate_angles)
        y = candidate_ranges * np.sin(candidate_angles)
        x_mean = float(np.average(x, weights=weights))
        y_mean = float(np.average(y, weights=weights))
        return float(math.atan2(y_mean, max(x_mean, self.min_valid_range)))

    def _path_clearance(self, path_angle, angles, ranges):
        if not self.use_footprint_clearance:
            return self.footprint_check_dist, False

        valid = ranges > self.min_valid_range
        if not np.any(valid):
            return self.footprint_check_dist, False

        check_dist = min(max(self.footprint_check_dist, self.vehicle_length), self.distance_cap)
        half_width = self.vehicle_width * 0.5 + max(self.footprint_safety_margin, 0.0)
        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        if not self.footprint_use_arc:
            return self._straight_path_clearance(path_angle, valid_angles, valid_ranges, check_dist, half_width)

        steering = float(np.clip(path_angle * self.target_point_scale, -self.max_steer, self.max_steer))
        curvature = math.tan(steering) / max(self.vehicle_wheelbase, 1e-3)
        if abs(curvature) < 1e-3:
            return self._straight_path_clearance(path_angle, valid_angles, valid_ranges, check_dist, half_width)

        sample_count = max(self.footprint_arc_samples, 6)
        front_start = min(self.vehicle_length * 0.5, check_dist)
        arc_s = np.linspace(front_start, check_dist, sample_count, dtype=np.float32)
        center_x = np.sin(curvature * arc_s) / curvature
        center_y = (1.0 - np.cos(curvature * arc_s)) / curvature

        point_x = valid_ranges * np.cos(valid_angles)
        point_y = valid_ranges * np.sin(valid_angles)
        ahead_mask = point_x > front_start
        if not np.any(ahead_mask):
            return check_dist, False
        point_x = point_x[ahead_mask]
        point_y = point_y[ahead_mask]

        dx = point_x[:, None] - center_x[None, :]
        dy = point_y[:, None] - center_y[None, :]
        dist_sq = dx * dx + dy * dy
        nearest_sample = np.argmin(dist_sq, axis=1)
        nearest_dist = np.sqrt(dist_sq[np.arange(dist_sq.shape[0]), nearest_sample])
        collision_mask = nearest_dist <= half_width
        if not np.any(collision_mask):
            return check_dist, False

        clearance = float(np.min(arc_s[nearest_sample[collision_mask]]))
        return clearance, True

    def _straight_path_clearance(self, path_angle, valid_angles, valid_ranges, check_dist, half_width):
        rel_angles = valid_angles - path_angle
        forward = valid_ranges * np.cos(rel_angles)
        lateral = valid_ranges * np.sin(rel_angles)
        front_start = min(self.vehicle_length * 0.5, check_dist)
        corridor_mask = (forward > front_start) & (forward <= check_dist) & (np.abs(lateral) <= half_width)
        if not np.any(corridor_mask):
            return check_dist, False
        return float(np.min(forward[corridor_mask])), True

    def _angle_side(self, angle):
        if angle > math.radians(2.0):
            return 1
        if angle < -math.radians(2.0):
            return -1
        return 0

    def _publish_visualization(self, result):
        stamp = self.get_clock().now().to_msg()
        angles = result['angles']
        self.gap_fov_pub.publish(
            self._make_fov_marker(stamp, angles, result['gap_mask'], 'gap_fov', 0, (0.1, 0.7, 1.0, 0.9))
        )
        self.nearest_pub.publish(self._make_nearest_marker(stamp, result))
        self.bubble_pub.publish(self._make_bubble_marker(stamp, result))
        self.gap_pub.publish(self._make_gap_marker(stamp, result))
        self.all_gaps_pub.publish(self._make_all_gaps_marker(stamp, result))
        self.gap_center_pub.publish(self._make_gap_center_marker(stamp, result))
        self.target_pub.publish(self._make_target_marker(stamp, result))
        self.footprint_pub.publish(self._make_footprint_marker(stamp))

    def _make_marker(self, stamp, marker_id, marker_type, scale, color, ns):
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        return marker

    def _polar_to_point(self, angle, radius, z=0.05):
        point = Point()
        point.x = float(radius * math.cos(angle))
        point.y = float(radius * math.sin(angle))
        point.z = float(z)
        return point

    def _make_fov_marker(self, stamp, angles, fov_mask, ns, marker_id, color):
        marker = self._make_marker(stamp, marker_id, Marker.LINE_LIST, (self.viz_fov_line_width, 0.0, 0.0), color, ns)
        indices = np.flatnonzero(fov_mask)
        if indices.size == 0:
            return marker
        start_angle = float(angles[indices[0]])
        end_angle = float(angles[indices[-1]])
        origin = Point(x=0.0, y=0.0, z=0.05)
        marker.points.extend([
            origin, self._polar_to_point(start_angle, self.distance_cap),
            origin, self._polar_to_point(end_angle, self.distance_cap),
        ])
        return marker

    def _make_nearest_marker(self, stamp, result):
        marker = self._make_marker(stamp, 2, Marker.SPHERE, (0.20, 0.20, 0.20), (1.0, 0.2, 0.2, 0.95), 'nearest')
        if result['nearest_idx'] is None or result['nearest_dist'] is None:
            marker.action = Marker.DELETE
            return marker
        marker.pose.position = self._polar_to_point(
            float(result['angles'][result['nearest_idx']]),
            float(result['nearest_dist']),
            z=0.08,
        )
        return marker

    def _make_bubble_marker(self, stamp, result):
        marker = self._make_marker(stamp, 3, Marker.LINE_LIST, (0.03, 0.0, 0.0), (1.0, 0.5, 0.0, 0.9), 'bubble')
        if not result['obstacle_clusters']:
            marker.action = Marker.DELETE
            return marker

        for cluster in result['obstacle_clusters']:
            obstacle_angle = float(result['angles'][cluster['center_idx']])
            obstacle_dist = float(cluster['dist'])
            half_angle = self._bubble_half_angle(obstacle_dist)
            center = self._polar_to_point(obstacle_angle, obstacle_dist)
            marker.points.extend([
                center, self._polar_to_point(obstacle_angle - half_angle, obstacle_dist),
                center, self._polar_to_point(obstacle_angle + half_angle, obstacle_dist),
            ])
        return marker

    def _make_gap_marker(self, stamp, result):
        marker = self._make_marker(
            stamp,
            4,
            Marker.POINTS,
            (self.viz_gap_endpoint_size, self.viz_gap_endpoint_size, self.viz_gap_endpoint_size),
            (0.2, 1.0, 0.2, 0.95),
            'gap',
        )
        if result['gap_start'] is None or result['gap_end'] is None:
            marker.action = Marker.DELETE
            return marker
        proc_ranges = result['proc_ranges']
        marker.points.append(self._polar_to_point(float(result['angles'][result['gap_start']]), float(proc_ranges[result['gap_start']]), z=0.06))
        marker.points.append(self._polar_to_point(float(result['angles'][result['gap_end']]), float(proc_ranges[result['gap_end']]), z=0.06))
        return marker

    def _make_all_gaps_marker(self, stamp, result):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.viz_frame
        delete_all.header.stamp = stamp
        delete_all.ns = 'all_gaps'
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        selected = result.get('selected_gap')
        proc_ranges = result['proc_ranges']
        for marker_id, gap in enumerate(result['gaps']):
            is_selected = (
                selected is not None
                and gap['start'] == selected['start']
                and gap['end'] == selected['end']
            )
            if is_selected:
                color = (0.2, 1.0, 0.2, 0.95)
            elif gap.get('driveable', False):
                color = (0.1, 0.7, 1.0, 0.65)
            else:
                color = (1.0, 0.15, 0.05, 0.45)
            marker = self._make_marker(
                stamp,
                marker_id,
                Marker.LINE_STRIP,
                (self.viz_all_gap_line_width, 0.0, 0.0),
                color,
                'all_gaps',
            )
            gap_len = gap['end'] - gap['start'] + 1
            step = max(1, gap_len // 30)
            for idx in range(gap['start'], gap['end'] + 1, step):
                marker.points.append(self._polar_to_point(float(result['angles'][idx]), float(proc_ranges[idx]), z=0.04))
            marker_array.markers.append(marker)
        return marker_array

    def _make_gap_center_marker(self, stamp, result):
        marker = self._make_marker(stamp, 5, Marker.SPHERE, (0.16, 0.16, 0.16), (0.1, 1.0, 1.0, 0.95), 'gap_center')
        if result['target_range'] <= 0.0:
            marker.action = Marker.DELETE
            return marker
        marker.pose.position = self._polar_to_point(
            float(result['gap_center_angle']),
            min(float(result['target_range']), self.max_range),
            z=0.07,
        )
        return marker

    def _make_target_marker(self, stamp, result):
        marker = self._make_marker(stamp, 6, Marker.ARROW, (0.06, 0.12, 0.18), (1.0, 1.0, 0.1, 0.95), 'target')
        if result['target_range'] <= 0.0:
            marker.action = Marker.DELETE
            return marker
        marker.points.append(Point(x=0.0, y=0.0, z=0.05))
        marker.points.append(self._polar_to_point(float(result['target_angle']), min(float(result['target_range']), self.max_range), z=0.05))
        return marker

    def _make_footprint_marker(self, stamp):
        marker = self._make_marker(stamp, 7, Marker.LINE_STRIP, (0.04, 0.0, 0.0), (0.9, 0.9, 0.9, 0.9), 'footprint')
        half_w = self.vehicle_width * 0.5
        rear_x = -self.vehicle_length * 0.5
        front_x = self.vehicle_length * 0.5
        marker.points.extend([
            Point(x=front_x, y=half_w, z=0.02),
            Point(x=front_x, y=-half_w, z=0.02),
            Point(x=rear_x, y=-half_w, z=0.02),
            Point(x=rear_x, y=half_w, z=0.02),
            Point(x=front_x, y=half_w, z=0.02),
        ])
        return marker


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
        try:
            node.drive_pub.publish(stop_msg)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
