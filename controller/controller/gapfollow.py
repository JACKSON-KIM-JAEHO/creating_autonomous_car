"""
Gap Following controller used by `reactive.launch.xml` when `mode:=gap`.

이 파일은 "Free Space / Gap" 기반 반응형 주행을 구현한다.
핵심 아이디어는 "장애물 자체를 따라가는 것"이 아니라, 전방에서 차량이
통과할 수 있는 빈 공간(gap)을 찾고 그 가운데를 향해 조향하는 것이다.

전체 흐름은 아래와 같다.
1. LiDAR ranges 전처리: NaN/inf 처리, 최대 거리 제한, 너무 작은 노이즈 제거
2. 좁은 전방 FOV에서 "실제로 위협이 되는" 가장 가까운 장애물 클러스터 선택
3. 선택된 장애물 주변에 bubble(안전 여유 구간)을 만들어 통과 후보에서 제거
4. 현재 속도/코너 여부를 보고 lookahead 거리를 정함
5. x = lookahead 평면에 도달 가능한 beam만 모아 "진짜 지나갈 수 있는 gap" 계산
6. gap 폭과 heading penalty를 이용해 가장 좋은 gap 하나를 선택
7. 선택된 gap의 중심을 target으로 삼아 steering 계산
8. steering 크기와 nearest obstacle 거리를 보고 속도를 자동 감속

언제 잘 맞는가:
- 맵 없이 LiDAR만으로 빠르게 반응해야 할 때
- 복도/트랙처럼 전방 빈 공간이 비교적 명확할 때
- 미리 만든 경로가 없어도 일단 충돌 회피 중심으로 달리고 싶을 때

주의할 점:
- gap-follow는 "최적 경로"를 고르는 알고리즘이 아니라, "지금 당장 지나가기
  쉬운 공간"을 고르는 알고리즘이다. 그래서 고속 주행이나 장기 계획이 필요한
  구간에서는 pure pursuit / MPC / waypoint planner와 역할을 나누는 것이 좋다.
- 급코너에서 lookahead가 너무 길면 코너 안쪽 벽을 늦게 인식해 바깥으로 밀릴 수 있다.
- bubble, safety margin, min_gap_beams를 너무 크게 잡으면 gap을 거의 못 찾아 멈출 수 있다.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class GapFollowNode(Node):
    """Lookahead 기반 true-gap follow node."""

    def __init__(self):
        super().__init__('gap_follow')

        # =========================
        # Core control params
        # =========================
        self.declare_parameter('control_rate_hz', 50.0)          # 제어 루프 주기. 너무 낮으면 반응이 늦고, 너무 높으면 노이즈를 더 자주 따라간다.

        # Driving
        self.declare_parameter('gf_speed', 2.0)                  # gap이 충분히 열려 있을 때 목표로 하는 기본 속도.
        self.declare_parameter('gf_min_speed', 0.6)              # 큰 조향이나 근접 장애물 상황에서도 유지할 최소 속도.
        self.declare_parameter('gf_max_steer', 0.4)              # 최종 steering saturation 한계. 차량 모델/서보 한계와 맞아야 한다.
        self.declare_parameter('gf_safe_distance', 2.5)          # nearest obstacle 감속 계산에서 "충분히 안전"하다고 보는 기준 거리.
        self.declare_parameter('gf_steer_speed_gain', 0.65)      # steering이 커질수록 속도를 얼마나 줄일지 결정하는 gain.

        # LiDAR preprocessing
        self.declare_parameter('gf_max_range', 10.0)             # inf를 대체하는 최대 range. sensor 실제 최대치보다 너무 작으면 열린 공간을 못 본다.
        self.declare_parameter('gf_distance_cap', 4.0)           # gap 판단에 사용할 추가 상한. 너무 크면 먼 opening에 끌리고, 너무 작으면 시야가 짧아진다.
        self.declare_parameter('gf_min_valid_range', 0.05)       # 이보다 작은 값은 노이즈/자기차체 근접값으로 보고 무효 처리.

        # Two FOVs
        self.declare_parameter('gf_gap_fov_deg', 150.0)          # 실제 gap 탐색에 사용하는 넓은 FOV. 코너 입구를 빨리 보려면 넓힌다.
        self.declare_parameter('gf_nearest_fov_deg', 50.0)       # nearest obstacle 탐색용 좁은 FOV. 너무 넓으면 옆 벽까지 nearest로 잡힌다.

        # Bubble
        self.declare_parameter('gf_bubble_radius', 0.30)         # nearest obstacle 주변에서 비워버릴 안전 반경.

        # Nearest obstacle cluster selection
        self.declare_parameter('gf_obstacle_trigger_dist', 1.8)  # 이 거리 안쪽의 beam만 "실제 위협" 후보로 본다.
        self.declare_parameter('gf_obstacle_cluster_tolerance', 0.20)  # 같은 obstacle cluster로 묶을 수 있는 range 변화 허용치.
        self.declare_parameter('gf_obstacle_cluster_min_beams', 3)      # cluster로 인정할 최소 연속 beam 수.
        self.declare_parameter('gf_obstacle_max_center_angle_deg', 18.0)  # 정면 근처 장애물만 bubble 중심 후보로 쓰기 위한 각도 제한.

        # Lookahead gap evaluation
        self.declare_parameter('gf_use_dynamic_lookahead', True)  # 속도가 빠를수록 lookahead를 조금 더 멀리 보는 옵션.
        self.declare_parameter('gf_lookahead_base', 1.4)          # 기본 lookahead 거리.
        self.declare_parameter('gf_lookahead_speed_gain', 0.6)    # 현재 속도에 따라 lookahead를 얼마나 늘릴지 결정.
        self.declare_parameter('gf_lookahead_min', 1.2)           # 너무 짧아져 과민 조향하지 않도록 하는 하한.
        self.declare_parameter('gf_lookahead_max', 2.5)           # 너무 길어져 코너 반응이 늦어지지 않도록 하는 상한.
        self.declare_parameter('gf_use_corner_lookahead', True)   # 급코너로 판단되면 별도의 짧은 lookahead를 쓰는 옵션.
        self.declare_parameter('gf_corner_lookahead', 0.85)       # 코너 모드에서 사용할 짧은 lookahead.
        self.declare_parameter('gf_corner_trigger_dist', 2.2)     # 정면 거리가 이보다 짧을 때만 코너 후보로 본다.
        self.declare_parameter('gf_corner_open_diff_dist', 0.8)   # 좌우 opening 차이가 이보다 커야 한쪽 코너로 판정.
        self.declare_parameter('gf_corner_side_angle_deg', 65.0)  # 좌우 opening을 샘플링할 중심 각도.
        self.declare_parameter('gf_corner_sector_width_deg', 20.0)  # opening 평균을 낼 sector 폭.

        # Gap validity
        self.declare_parameter('gf_min_gap_beams', 6)             # beam 개수가 너무 적은 gap을 제거하기 위한 최소 개수.
        self.declare_parameter('gf_vehicle_safety_margin', 0.10)  # 차량 폭 양옆에 추가로 요구할 여유 폭.

        # Gap scoring
        self.declare_parameter('gf_gap_heading_weight', 0.15)     # gap 중심 각도가 클수록 주는 penalty. 너무 크면 코너 진입이 늦어진다.
        self.declare_parameter('gf_forward_gap_bonus', 0.8)       # 정면(y=0)을 포함한 gap에 주는 bonus. 직진 안정성 확보용.

        # Target / steering smoothing
        self.declare_parameter('gf_target_point_scale', 1.0)      # target angle -> steering 변환 계수. 조향 민감도를 바꿀 때 사용.
        self.declare_parameter('gf_use_steer_smoothing', True)    # 이전 steering을 섞어 조향 jitter를 줄일지 여부.
        self.declare_parameter('gf_steer_smoothing_alpha', 0.75)  # 1에 가까울수록 이전 steering을 더 오래 유지한다.

        # Vehicle geometry
        self.declare_parameter('vehicle_width', 0.30)             # gap 통과 가능 여부를 판단할 실제 차량 폭.
        self.declare_parameter('vehicle_length', 0.50)            # RViz footprint에 표시할 차량 길이.

        # Debug / Visualization
        self.declare_parameter('viz_frame', 'base_link')          # RViz marker 기준 프레임.
        self.declare_parameter('viz_gap_fov_topic', '/gf/gap_fov_marker')
        self.declare_parameter('viz_nearest_fov_topic', '/gf/nearest_fov_marker')
        self.declare_parameter('viz_nearest_point_topic', '/gf/nearest_point_marker')
        self.declare_parameter('viz_bubble_topic', '/gf/bubble_marker')
        self.declare_parameter('viz_gap_topic', '/gf/gap_marker')
        self.declare_parameter('viz_gap_center_topic', '/gf/gap_center_marker')
        self.declare_parameter('viz_target_topic', '/gf/target_marker')
        self.declare_parameter('viz_footprint_topic', '/gf/footprint_marker')
        self.declare_parameter('viz_fov_line_width', 0.05)        # FOV 경계선 시각화 두께.
        self.declare_parameter('viz_gap_endpoint_size', 0.18)     # gap 시작/끝 점 marker 크기.
        self.declare_parameter('debug_log_every_n', 10)           # 몇 주기마다 debug log를 남길지 결정.

        p = lambda name: self.get_parameter(name).value

        # =========================
        # Load params
        # =========================
        # 선언과 로드는 분리해 두면, 위쪽에서 "무슨 파라미터가 있는지" 한눈에 보고
        # 아래쪽에서 "실행 시 어떤 멤버 변수에 연결되는지" 쉽게 추적할 수 있다.
        self.control_rate_hz = float(p('control_rate_hz'))

        self.nominal_speed = float(p('gf_speed'))
        self.min_speed = float(p('gf_min_speed'))
        self.max_steer = float(p('gf_max_steer'))
        self.safe_distance = float(p('gf_safe_distance'))
        self.steer_speed_gain = float(p('gf_steer_speed_gain'))

        self.max_range = float(p('gf_max_range'))
        self.distance_cap = float(p('gf_distance_cap'))
        self.min_valid_range = float(p('gf_min_valid_range'))

        self.gap_fov_deg = float(p('gf_gap_fov_deg'))
        self.nearest_fov_deg = float(p('gf_nearest_fov_deg'))

        self.bubble_radius = float(p('gf_bubble_radius'))

        self.obstacle_trigger_dist = float(p('gf_obstacle_trigger_dist'))
        self.obstacle_cluster_tolerance = float(p('gf_obstacle_cluster_tolerance'))
        self.obstacle_cluster_min_beams = int(p('gf_obstacle_cluster_min_beams'))
        self.obstacle_max_center_angle_deg = float(p('gf_obstacle_max_center_angle_deg'))

        self.use_dynamic_lookahead = bool(p('gf_use_dynamic_lookahead'))
        self.lookahead_base = float(p('gf_lookahead_base'))
        self.lookahead_speed_gain = float(p('gf_lookahead_speed_gain'))
        self.lookahead_min = float(p('gf_lookahead_min'))
        self.lookahead_max = float(p('gf_lookahead_max'))
        self.use_corner_lookahead = bool(p('gf_use_corner_lookahead'))
        self.corner_lookahead = float(p('gf_corner_lookahead'))
        self.corner_trigger_dist = float(p('gf_corner_trigger_dist'))
        self.corner_open_diff_dist = float(p('gf_corner_open_diff_dist'))
        self.corner_side_angle = math.radians(float(p('gf_corner_side_angle_deg')))
        self.corner_sector_width = math.radians(float(p('gf_corner_sector_width_deg')))

        self.min_gap_beams = int(p('gf_min_gap_beams'))
        self.vehicle_safety_margin = float(p('gf_vehicle_safety_margin'))

        self.gap_heading_weight = float(p('gf_gap_heading_weight'))
        self.forward_gap_bonus = float(p('gf_forward_gap_bonus'))

        self.target_point_scale = float(p('gf_target_point_scale'))
        self.use_steer_smoothing = bool(p('gf_use_steer_smoothing'))
        self.steer_smoothing_alpha = float(p('gf_steer_smoothing_alpha'))

        self.vehicle_width = float(p('vehicle_width'))
        self.vehicle_length = float(p('vehicle_length'))

        self.viz_frame = str(p('viz_frame'))
        self.viz_gap_fov_topic = str(p('viz_gap_fov_topic'))
        self.viz_nearest_fov_topic = str(p('viz_nearest_fov_topic'))
        self.viz_nearest_point_topic = str(p('viz_nearest_point_topic'))
        self.viz_bubble_topic = str(p('viz_bubble_topic'))
        self.viz_gap_topic = str(p('viz_gap_topic'))
        self.viz_gap_center_topic = str(p('viz_gap_center_topic'))
        self.viz_target_topic = str(p('viz_target_topic'))
        self.viz_footprint_topic = str(p('viz_footprint_topic'))
        self.viz_fov_line_width = float(p('viz_fov_line_width'))
        self.viz_gap_endpoint_size = float(p('viz_gap_endpoint_size'))
        self.debug_log_every_n = int(p('debug_log_every_n'))

        # =========================
        # State
        # =========================
        self.scan: Optional[LaserScan] = None
        self.odom: Optional[Odometry] = None
        self.prev_steering: float = 0.0  # steering smoothing에 사용되는 직전 조향값
        self.debug_count: int = 0        # 과도한 로그 스팸을 막기 위한 카운터

        # =========================
        # ROS interfaces
        # =========================
        # 입력은 LiDAR + odom 두 가지만 받는다.
        # odom은 gap 탐색 자체보다는 dynamic lookahead 계산에 쓰인다.
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)

        # 출력은 Ackermann steering/speed command.
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            10
        )

        # 아래 marker들은 "왜 이 steering이 나왔는지" 설명 가능한 디버깅용이다.
        self.gap_fov_pub = self.create_publisher(Marker, self.viz_gap_fov_topic, 10)
        self.nearest_fov_pub = self.create_publisher(Marker, self.viz_nearest_fov_topic, 10)
        self.nearest_pub = self.create_publisher(Marker, self.viz_nearest_point_topic, 10)
        self.bubble_pub = self.create_publisher(Marker, self.viz_bubble_topic, 10)
        self.gap_pub = self.create_publisher(Marker, self.viz_gap_topic, 10)
        self.gap_center_pub = self.create_publisher(Marker, self.viz_gap_center_topic, 10)
        self.target_pub = self.create_publisher(Marker, self.viz_target_topic, 10)
        self.footprint_pub = self.create_publisher(Marker, self.viz_footprint_topic, 10)

        self.create_timer(1.0 / self.control_rate_hz, self._loop)

        self.get_logger().info('GapFollowNode ready (lookahead-based true gap FTG)')

    # =========================================================
    # Callbacks
    # =========================================================
    def _scan_cb(self, msg: LaserScan):
        """가장 최근 LiDAR scan을 저장한다."""
        self.scan = msg

    def _odom_cb(self, msg: Odometry):
        """현재 속도 계산용 odom을 저장한다."""
        self.odom = msg

    # =========================================================
    # Main loop
    # =========================================================
    def _loop(self):
        # scan이 아직 없으면 계산할 수 있는 입력이 없으므로 대기한다.
        if self.scan is None:
            return

        result = self._compute()

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = float(result['steering'])
        drive_msg.drive.speed = float(result['speed'])
        self.drive_pub.publish(drive_msg)

        # 항상 visualization을 publish하므로, RViz에서 현재 판단 과정을 매 주기 확인할 수 있다.
        self._publish_visualization(result)

    # =========================================================
    # FTG core
    # =========================================================
    def _compute(self) -> Dict[str, Any]:
        """한 프레임의 scan으로 steering/speed를 계산한다."""
        ranges = np.array(self.scan.ranges, dtype=np.float32)

        # 1) LiDAR 전처리
        # NaN은 없는 값으로, inf는 "매우 멀리 열려 있음"으로 간주한다.
        # 그 뒤 max_range와 distance_cap을 차례대로 적용해 지나치게 먼 값이
        # 알고리즘을 흔들지 않도록 한다.
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)
        ranges = np.clip(ranges, 0.0, self.distance_cap)

        n = len(ranges)
        angles = self.scan.angle_min + np.arange(n, dtype=np.float32) * self.scan.angle_increment

        gap_half_fov = math.radians(self.gap_fov_deg) * 0.5
        nearest_half_fov = math.radians(self.nearest_fov_deg) * 0.5

        # 넓은 FOV는 "gap 찾기" 용도, 좁은 FOV는 "지금 제일 위험한 전방 장애물" 용도다.
        # 둘을 분리하면 옆 벽 때문에 bubble이 과도하게 생기는 현상을 줄일 수 있다.
        gap_mask = np.abs(angles) <= gap_half_fov
        nearest_mask = np.abs(angles) <= nearest_half_fov

        gap_ranges = np.where(gap_mask, ranges, 0.0)
        nearest_ranges = np.where(nearest_mask, ranges, 0.0)

        # 너무 작은 값은 차체 반사/센서 노이즈일 가능성이 높으므로 제거한다.
        gap_ranges[gap_ranges < self.min_valid_range] = 0.0
        nearest_ranges[nearest_ranges < self.min_valid_range] = 0.0

        # 2) nearest obstacle selection
        # 전방 좁은 FOV 안에서 실제 cluster 형태를 가진 장애물만 선택한다.
        # 단일 beam 노이즈나 너무 옆에 있는 벽을 nearest로 쓰지 않기 위함이다.
        nearest_idx, nearest_dist = self._select_nearest_obstacle(nearest_ranges, angles)

        nearest_angle = None
        if nearest_idx is not None and nearest_dist is not None:
            nearest_angle = float(angles[nearest_idx])

        # 3) bubble 적용
        # nearest obstacle 주변은 실제 차량 폭과 오차를 고려해 지나가지 않을
        # 영역으로 지워준다. 이 bubble이 너무 작으면 스치듯 통과하려 하고,
        # 너무 크면 실제로 가능한 gap도 막아버린다.
        proc_ranges = gap_ranges.copy()
        bubble_mask = np.zeros_like(gap_mask, dtype=bool)

        if nearest_idx is not None and nearest_dist is not None:
            bubble_half_angle = self._bubble_half_angle(nearest_dist)
            bubble_mask = np.abs(angles - nearest_angle) <= bubble_half_angle
            bubble_mask &= gap_mask
            proc_ranges[bubble_mask] = 0.0

        # 4) lookahead 선택
        # lookahead는 "어느 x 위치에서 통과 가능한지를 보겠다"는 기준이다.
        # 짧으면 민첩하지만 흔들리고, 길면 부드럽지만 급코너 반응이 늦다.
        lookahead_dist, lookahead_mode = self._compute_lookahead_distance(ranges, angles)

        # distance_cap보다 먼 곳을 lookahead 평면으로 두면, 실제로는 도달 불가능한
        # beam을 gap처럼 오해할 수 있으므로 안전하게 한 번 더 clamp한다.
        lookahead_dist = min(lookahead_dist, max(self.min_valid_range, self.distance_cap - 0.05))
        lookahead_dist = max(lookahead_dist, self.lookahead_min)

        # 5) true gap 구성
        # 단순히 연속된 non-zero beam을 gap으로 보는 게 아니라, x = lookahead 평면에
        # 실제로 닿는 beam만 사용해서 "그 평면에서 차량이 통과할 폭이 있는가?"를 계산한다.
        gaps = self._find_true_gaps(proc_ranges, angles, lookahead_dist)
        selected_gap = self._select_gap(gaps)

        # gap이 하나도 없으면 무리하게 꺾지 않고 정지하는 보수적 fallback을 사용한다.
        if selected_gap is None:
            return self._fallback_result(
                angles=angles,
                gap_mask=gap_mask,
                nearest_mask=nearest_mask,
                nearest_idx=nearest_idx,
                nearest_dist=nearest_dist,
                bubble_mask=bubble_mask,
                lookahead_dist=lookahead_dist,
            )

        # 6) target = 선택된 gap의 중앙
        # 현재 구현은 가장 복잡한 target optimization 대신, 선택된 gap의 중심을 향한다.
        # 구조가 단순해 디버깅이 쉽고, 주행 의도가 marker로도 직관적으로 보인다.
        target_y = 0.5 * (selected_gap['y_min'] + selected_gap['y_max'])
        target_angle = float(math.atan2(target_y, lookahead_dist))
        target_range = float(lookahead_dist / max(math.cos(target_angle), 1e-3))
        target_idx = int(np.argmin(np.abs(angles - target_angle)))

        gap_center_angle = float(selected_gap['center_angle'])
        gap_center_range = float(lookahead_dist / max(math.cos(gap_center_angle), 1e-3))

        # 7) steering 계산
        # target angle을 steering 명령으로 쓰되, 차량 한계를 넘지 않게 clamp한다.
        raw_steering = float(np.clip(
            target_angle * self.target_point_scale,
            -self.max_steer,
            self.max_steer
        ))

        steering = raw_steering
        if self.use_steer_smoothing:
            # smoothing은 급격한 좌우 스위칭을 줄여주지만, 너무 크면 코너 반응이 늦어진다.
            alpha = float(np.clip(self.steer_smoothing_alpha, 0.0, 1.0))
            steering = alpha * self.prev_steering + (1.0 - alpha) * raw_steering

        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        self.prev_steering = steering

        # 8) 속도 계산
        # 큰 조향이 필요할수록, 그리고 nearest obstacle이 가까울수록 감속한다.
        # 즉 "커브가 급하거나 공간이 좁으면 느리게"라는 보수적인 정책이다.
        steer_ratio = abs(steering) / self.max_steer if self.max_steer > 1e-6 else 0.0
        clearance_dist = nearest_dist if nearest_dist is not None else self.safe_distance
        clearance_ratio = min(clearance_dist / self.safe_distance, 1.0)

        speed = self.nominal_speed * (1.0 - self.steer_speed_gain * steer_ratio)
        speed *= 0.5 + 0.5 * clearance_ratio
        speed = float(np.clip(speed, self.min_speed, self.nominal_speed))

        self.debug_count += 1
        if self.debug_count % max(1, self.debug_log_every_n) == 0:
            nearest_text = 'none'
            if nearest_idx is not None and nearest_dist is not None:
                nearest_text = f'{nearest_idx}/{nearest_dist:.2f}'

            self.get_logger().info(
                f'[GAP] nearest={nearest_text} '
                f'gap={selected_gap["start"]}-{selected_gap["end"]} '
                f'width={selected_gap["width"]:.2f} '
                f'target={target_idx}/{math.degrees(target_angle):.1f}deg '
                f'steer={steering:.3f} speed={speed:.3f} '
                f'lookahead={lookahead_dist:.2f}/{lookahead_mode}'
            )

        return {
            'steering': steering,
            'speed': speed,
            'angles': angles,
            'proc_ranges': proc_ranges,
            'gap_mask': gap_mask,
            'nearest_mask': nearest_mask,
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'bubble_mask': bubble_mask,
            'gap_start': int(selected_gap['start']),
            'gap_end': int(selected_gap['end']),
            'target_idx': target_idx,
            'target_range': target_range,
            'target_angle': target_angle,
            'gap_center_angle': gap_center_angle,
            'gap_center_range': gap_center_range,
            'gap_width': float(selected_gap['width']),
            'lookahead_dist': lookahead_dist,
            'lookahead_mode': lookahead_mode,
            'used_fallback': False,
        }

    # =========================================================
    # Helper: nearest obstacle selection
    # =========================================================
    def _select_nearest_obstacle(
        self,
        nearest_ranges: np.ndarray,
        angles: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        bubble 중심으로 사용할 전방 obstacle 하나를 고른다.

        선택 기준:
        - 너무 먼 beam 제외
        - 연속된 beam cluster만 인정
        - cluster 내부 거리 퍼짐이 너무 크면 제외
        - 정면 근처 obstacle만 인정
        - 그중 평균 거리가 가장 가까운 cluster를 선택
        """
        candidate_mask = (
            (nearest_ranges > 0.0) &
            (nearest_ranges <= self.obstacle_trigger_dist)
        )

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return None, None

        # 인접 index끼리 묶어 연속된 beam cluster를 만든다.
        splits = np.split(candidate_indices, np.where(np.diff(candidate_indices) > 1)[0] + 1)

        best_cluster = None
        best_cluster_mean = None

        for cluster in splits:
            if len(cluster) < self.obstacle_cluster_min_beams:
                continue

            cluster_ranges = nearest_ranges[cluster]

            # 하나의 물체라면 cluster 내부 거리 변화가 너무 크지 않아야 한다.
            if float(np.max(cluster_ranges) - np.min(cluster_ranges)) > self.obstacle_cluster_tolerance:
                continue

            center_idx = int(cluster[len(cluster) // 2])
            center_angle_deg = abs(math.degrees(float(angles[center_idx])))
            if center_angle_deg > self.obstacle_max_center_angle_deg:
                continue

            mean_dist = float(np.mean(cluster_ranges))

            if best_cluster_mean is None or mean_dist < best_cluster_mean:
                best_cluster = cluster
                best_cluster_mean = mean_dist

        if best_cluster is None:
            return None, None

        center_idx = int(best_cluster[len(best_cluster) // 2])
        center_dist = float(np.mean(nearest_ranges[best_cluster]))
        return center_idx, center_dist

    # =========================================================
    # Helper: bubble
    # =========================================================
    def _bubble_half_angle(self, distance: float) -> float:
        """
        bubble 반경(m)을 각도(rad)로 바꾼다.

        obstacle이 가까울수록 같은 반경도 더 큰 각도를 차지하므로,
        가까운 장애물 앞에서는 bubble이 더 넓게 보이게 된다.
        """
        distance = max(distance, self.min_valid_range)
        ratio = np.clip(self.bubble_radius / distance, -1.0, 1.0)
        return float(math.asin(ratio))

    # =========================================================
    # Helper: lookahead
    # =========================================================
    def _compute_current_speed(self) -> float:
        """odom으로부터 현재 평면 속도를 계산한다."""
        if self.odom is None:
            return 0.0

        vx = float(self.odom.twist.twist.linear.x)
        vy = float(self.odom.twist.twist.linear.y)
        return float(math.sqrt(vx * vx + vy * vy))

    def _compute_lookahead_distance(self, ranges: np.ndarray, angles: np.ndarray) -> Tuple[float, str]:
        """
        lookahead 거리와 그 선택 모드를 반환한다.

        우선순위:
        1. 급코너가 감지되면 corner lookahead 사용
        2. 아니고 dynamic lookahead가 꺼져 있으면 fixed
        3. 아니면 현재 속도 기반 dynamic
        """
        if self.use_corner_lookahead:
            corner_mode = self._detect_corner_for_lookahead(ranges, angles)
            if corner_mode != 'none':
                lookahead = float(np.clip(
                    self.corner_lookahead,
                    self.lookahead_min,
                    self.lookahead_max
                ))
                return lookahead, corner_mode

        if not self.use_dynamic_lookahead:
            lookahead = float(np.clip(self.lookahead_base, self.lookahead_min, self.lookahead_max))
            return lookahead, 'fixed'

        current_speed = self._compute_current_speed()
        lookahead = self.lookahead_base + self.lookahead_speed_gain * current_speed
        lookahead = float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))
        return lookahead, 'dynamic'

    def _detect_corner_for_lookahead(self, ranges: np.ndarray, angles: np.ndarray) -> str:
        """
        정면이 막혀 있고 좌/우 한쪽 opening이 확실히 더 크면 급코너로 본다.

        이 판단이 필요한 이유:
        - 일반 dynamic lookahead는 속도가 빠를수록 길어지는 경향이 있다.
        - 그런데 급코너에서는 오히려 lookahead를 짧게 해야 안쪽 회전이 빨라진다.
        """
        front_dist = self._range_at_angle(ranges, angles, 0.0)
        if front_dist > self.corner_trigger_dist:
            return 'none'

        half_width = self.corner_sector_width * 0.5
        left_open = self._sector_mean(
            ranges,
            angles,
            self.corner_side_angle - half_width,
            self.corner_side_angle + half_width,
        )
        right_open = self._sector_mean(
            ranges,
            angles,
            -self.corner_side_angle - half_width,
            -self.corner_side_angle + half_width,
        )

        if left_open - right_open > self.corner_open_diff_dist:
            return 'corner_left'
        if right_open - left_open > self.corner_open_diff_dist:
            return 'corner_right'
        return 'none'

    def _range_at_angle(self, ranges: np.ndarray, angles: np.ndarray, target_angle: float) -> float:
        """특정 각도와 가장 가까운 beam의 range를 가져온다."""
        if ranges.size == 0:
            return self.distance_cap

        idx = int(np.argmin(np.abs(angles - target_angle)))
        value = float(ranges[idx])
        return self.distance_cap if value <= self.min_valid_range else value

    def _sector_mean(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        start_angle: float,
        end_angle: float
    ) -> float:
        """특정 각도 구간의 평균 opening을 계산한다."""
        lo = min(start_angle, end_angle)
        hi = max(start_angle, end_angle)
        mask = (angles >= lo) & (angles <= hi) & (ranges > self.min_valid_range)
        if not np.any(mask):
            return 0.0
        return float(np.mean(ranges[mask]))

    # =========================================================
    # Helper: true gap finding on lookahead plane
    # =========================================================
    def _find_true_gaps(
        self,
        proc_ranges: np.ndarray,
        angles: np.ndarray,
        lookahead_dist: float
    ) -> List[Dict[str, Any]]:
        """
        x = lookahead_dist 평면에서 실제로 통과 가능한 gap만 찾는다.

        핵심 포인트:
        - beam이 충분히 멀리 뻗어 x = lookahead 평면까지 도달해야 valid
        - valid beam들을 연속 cluster로 묶은 뒤
        - 그 평면에서의 y 폭(width)이 차량 폭 + safety margin보다 커야 통과 가능
        """
        # beam 끝점의 x 성분. x가 lookahead보다 짧으면 해당 beam은 그 평면까지 닿지 못한다.
        x_reach = proc_ranges * np.cos(angles)
        valid_mask = (proc_ranges > 0.0) & (x_reach >= lookahead_dist)

        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size == 0:
            return []

        splits = np.split(valid_idx, np.where(np.diff(valid_idx) > 1)[0] + 1)

        gaps: List[Dict[str, Any]] = []
        required_width = self.vehicle_width + 2.0 * self.vehicle_safety_margin

        for cluster in splits:
            if len(cluster) < self.min_gap_beams:
                continue

            cluster_angles = angles[cluster]

            # x = lookahead 평면에서 각 beam이 만나는 y 좌표.
            # y = x * tan(theta), 여기서는 x = lookahead_dist.
            y_hits = lookahead_dist * np.tan(cluster_angles)

            y_min = float(np.min(y_hits))
            y_max = float(np.max(y_hits))
            width = float(y_max - y_min)

            # 폭이 차량 폭 + margin보다 작으면 "보이긴 하지만 실제로는 못 지나감"으로 본다.
            if width < required_width:
                continue

            center_y = 0.5 * (y_min + y_max)
            center_angle = float(math.atan2(center_y, lookahead_dist))

            gaps.append({
                'start': int(cluster[0]),
                'end': int(cluster[-1]),
                'count': int(len(cluster)),
                'y_min': y_min,
                'y_max': y_max,
                'width': width,
                'center_angle': center_angle,
            })

        return gaps

    def _select_gap(self, gaps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        gap 폭을 기본 점수로 쓰고, heading penalty와 forward bonus를 적용해 최종 선택한다.

        현재 구현은 일부러 단순하다.
        - 폭이 넓을수록 좋다.
        - 너무 큰 heading change가 필요한 gap은 약간 불리하다.
        - 정면을 포함하는 gap은 직진 안정성을 위해 bonus를 준다.
        """
        if not gaps:
            return None

        best_gap = None
        best_score = None

        for gap in gaps:
            score = gap['width']
            score -= self.gap_heading_weight * abs(math.degrees(gap['center_angle']))

            if gap['y_min'] <= 0.0 <= gap['y_max']:
                score += self.forward_gap_bonus

            if best_score is None or score > best_score:
                best_gap = gap
                best_score = score

        return best_gap

    # =========================================================
    # Fallback
    # =========================================================
    def _fallback_result(
        self,
        angles: np.ndarray,
        gap_mask: np.ndarray,
        nearest_mask: np.ndarray,
        nearest_idx: Optional[int] = None,
        nearest_dist: Optional[float] = None,
        bubble_mask: Optional[np.ndarray] = None,
        lookahead_dist: float = 0.0,
    ) -> Dict[str, Any]:
        """
        gap을 찾지 못했을 때의 보수적 처리.

        현재 정책:
        - steering = 0
        - speed = 0

        즉, "애매하면 멈춘다" 쪽에 가깝다.
        좁은 실내나 복잡한 장애물 환경에서는 안전하지만, 지나치게 보수적이라고 느껴지면
        bubble/safety margin/min_gap_beams 쪽을 먼저 조정하는 것이 좋다.
        """
        center_candidates = np.flatnonzero(gap_mask)
        target_idx = int(center_candidates[len(center_candidates) // 2]) if center_candidates.size else 0
        target_angle = float(angles[target_idx]) if len(angles) > 0 else 0.0

        self.prev_steering = 0.0

        self.debug_count += 1
        if self.debug_count % max(1, self.debug_log_every_n) == 0:
            nearest_text = 'none'
            if nearest_idx is not None and nearest_dist is not None:
                nearest_text = f'{nearest_idx}/{nearest_dist:.2f}'

            self.get_logger().info(
                f'[GAP] nearest={nearest_text} fallback=yes '
                f'target={target_idx}/{math.degrees(target_angle):.1f}deg '
                f'steer=0.000 speed=0.000 lookahead={lookahead_dist:.2f}'
            )

        return {
            'steering': 0.0,
            'speed': 0.0,
            'angles': angles,
            'proc_ranges': np.zeros_like(angles, dtype=np.float32),
            'gap_mask': gap_mask,
            'nearest_mask': nearest_mask,
            'nearest_idx': nearest_idx,
            'nearest_dist': nearest_dist,
            'bubble_mask': bubble_mask if bubble_mask is not None else np.zeros_like(gap_mask, dtype=bool),
            'gap_start': None,
            'gap_end': None,
            'target_idx': target_idx,
            'target_range': 0.0,
            'target_angle': target_angle,
            'gap_center_angle': target_angle,
            'gap_center_range': 0.0,
            'gap_width': 0.0,
            'lookahead_dist': lookahead_dist,
            'used_fallback': True,
        }

    # =========================================================
    # Visualization helpers
    # =========================================================
    def _publish_visualization(self, result: Dict[str, Any]):
        """현재 gap-follow 판단 과정을 RViz marker로 내보낸다."""
        stamp = self.get_clock().now().to_msg()
        angles = result['angles']

        self.gap_fov_pub.publish(
            self._make_fov_marker(
                stamp, angles, result['gap_mask'],
                ns='gap_fov', marker_id=0,
                color=(0.1, 0.7, 1.0, 0.9)
            )
        )

        self.nearest_fov_pub.publish(
            self._make_fov_marker(
                stamp, angles, result['nearest_mask'],
                ns='nearest_fov', marker_id=1,
                color=(1.0, 0.2, 1.0, 0.9)
            )
        )

        self.nearest_pub.publish(self._make_nearest_marker(stamp, result))
        self.bubble_pub.publish(self._make_bubble_marker(stamp, result))
        self.gap_pub.publish(self._make_gap_marker(stamp, result))
        self.gap_center_pub.publish(self._make_gap_center_marker(stamp, result))
        self.target_pub.publish(self._make_target_marker(stamp, result))
        self.footprint_pub.publish(self._make_footprint_marker(stamp))

    def _make_marker(
        self,
        stamp,
        marker_id: int,
        marker_type: int,
        scale,
        color,
        ns: str
    ) -> Marker:
        """공통 marker 생성 로직."""
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale[0])
        marker.scale.y = float(scale[1])
        marker.scale.z = float(scale[2])
        marker.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=float(color[3]))
        return marker

    def _polar_to_point(self, angle: float, radius: float, z: float = 0.05) -> Point:
        """polar 좌표를 RViz marker용 Point로 변환한다."""
        point = Point()
        point.x = float(radius * math.cos(angle))
        point.y = float(radius * math.sin(angle))
        point.z = float(z)
        return point

    def _make_fov_marker(self, stamp, angles, fov_mask, ns, marker_id, color):
        """FOV 경계를 선 두 개로 보여준다."""
        marker = self._make_marker(
            stamp, marker_id, Marker.LINE_LIST,
            (self.viz_fov_line_width, 0.0, 0.0),
            color, ns
        )

        indices = np.flatnonzero(fov_mask)
        if indices.size == 0:
            return marker

        start_angle = float(angles[indices[0]])
        end_angle = float(angles[indices[-1]])

        origin = Point(x=0.0, y=0.0, z=0.05)
        marker.points.extend([
            origin, self._polar_to_point(start_angle, self.max_range),
            origin, self._polar_to_point(end_angle, self.max_range),
        ])
        return marker

    def _make_nearest_marker(self, stamp, result):
        """선택된 nearest obstacle 위치를 구체로 표시한다."""
        marker = self._make_marker(
            stamp, 10, Marker.SPHERE,
            (0.20, 0.20, 0.20),
            (1.0, 0.2, 0.2, 0.95),
            'nearest'
        )

        nearest_idx = result['nearest_idx']
        if nearest_idx is None or result['nearest_dist'] is None:
            marker.action = Marker.DELETE
            return marker

        marker.pose.position = self._polar_to_point(
            float(result['angles'][nearest_idx]),
            float(result['nearest_dist']),
            z=0.08
        )
        return marker

    def _make_bubble_marker(self, stamp, result):
        """nearest obstacle 주변 bubble 각도 범위를 표시한다."""
        marker = self._make_marker(
            stamp, 11, Marker.LINE_LIST,
            (0.03, 0.0, 0.0),
            (1.0, 0.5, 0.0, 0.9),
            'bubble'
        )

        nearest_idx = result['nearest_idx']
        if nearest_idx is None or result['nearest_dist'] is None:
            marker.action = Marker.DELETE
            return marker

        angles = result['angles']
        nearest_angle = float(angles[nearest_idx])
        half_angle = self._bubble_half_angle(float(result['nearest_dist']))
        left_angle = nearest_angle - half_angle
        right_angle = nearest_angle + half_angle
        center = self._polar_to_point(nearest_angle, float(result['nearest_dist']))

        marker.points.extend([
            center, self._polar_to_point(left_angle, float(result['nearest_dist'])),
            center, self._polar_to_point(right_angle, float(result['nearest_dist'])),
        ])
        return marker

    def _make_gap_marker(self, stamp, result):
        """선택된 gap의 양 끝점을 표시한다."""
        marker = self._make_marker(
            stamp, 12, Marker.POINTS,
            (self.viz_gap_endpoint_size, self.viz_gap_endpoint_size, self.viz_gap_endpoint_size),
            (0.2, 1.0, 0.2, 0.95),
            'gap'
        )

        if result['gap_start'] is None or result['gap_end'] is None:
            marker.action = Marker.DELETE
            return marker

        angles = result['angles']
        proc_ranges = result['proc_ranges']

        start_range = float(max(proc_ranges[result['gap_start']], self.min_valid_range))
        end_range = float(max(proc_ranges[result['gap_end']], self.min_valid_range))

        marker.points.append(self._polar_to_point(float(angles[result['gap_start']]), start_range, z=0.06))
        marker.points.append(self._polar_to_point(float(angles[result['gap_end']]), end_range, z=0.06))
        return marker

    def _make_gap_center_marker(self, stamp, result):
        """선택된 gap의 중심점을 표시한다."""
        marker = self._make_marker(
            stamp, 13, Marker.SPHERE,
            (0.16, 0.16, 0.16),
            (0.1, 1.0, 1.0, 0.95),
            'gap_center'
        )

        gap_center_range = float(result['gap_center_range'])
        if gap_center_range <= 0.0:
            marker.action = Marker.DELETE
            return marker

        marker.pose.position = self._polar_to_point(
            float(result['gap_center_angle']),
            min(gap_center_range, self.max_range),
            z=0.07
        )
        return marker

    def _make_target_marker(self, stamp, result):
        """최종 target direction을 화살표로 표시한다."""
        marker = self._make_marker(
            stamp, 14, Marker.ARROW,
            (0.06, 0.12, 0.18),
            (1.0, 1.0, 0.1, 0.95),
            'target'
        )

        target_range = float(result['target_range'])
        if target_range <= 0.0:
            marker.action = Marker.DELETE
            return marker

        marker.points.append(Point(x=0.0, y=0.0, z=0.05))
        marker.points.append(
            self._polar_to_point(
                float(result['target_angle']),
                min(target_range, self.max_range),
                z=0.05
            )
        )
        return marker

    def _make_footprint_marker(self, stamp):
        """현재 gap 폭 판단 기준이 되는 차량 footprint를 표시한다."""
        marker = self._make_marker(
            stamp, 15, Marker.LINE_STRIP,
            (0.04, 0.0, 0.0),
            (0.9, 0.9, 0.9, 0.9),
            'footprint'
        )

        half_w = self.vehicle_width * 0.5
        rear_x = -self.vehicle_length * 0.5
        front_x = self.vehicle_length * 0.5

        corners = [
            Point(x=front_x, y=half_w, z=0.02),
            Point(x=front_x, y=-half_w, z=0.02),
            Point(x=rear_x, y=-half_w, z=0.02),
            Point(x=rear_x, y=half_w, z=0.02),
            Point(x=front_x, y=half_w, z=0.02),
        ]
        marker.points.extend(corners)
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 종료 시 마지막으로 정지 명령을 한 번 보내서 차량이 명령 없이
        # 계속 움직이는 상황을 줄인다.
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
