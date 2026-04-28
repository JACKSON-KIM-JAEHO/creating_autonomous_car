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




class GapFollowNode(Node):
    """
    Follow-the-Gap 알고리즘 기반 자율 주행 노드.

    동작 흐름:
      1. LiDAR 스캔 전처리 (NaN/Inf 제거, 범위 제한, FOV 마스킹)
      2. 전방 거리 기반 사전 감속 속도 계산
      3. 커브 힌트 감지 (정면 vs 측면 openness 비교)
      4. 가장 가까운 장애물에 Safety Bubble 적용 (선택)
      5. 최대 Gap 탐색 및 목표 지점 선택
      6. PID 조향 제어 + 직선/커브 보정
      7. 속도 결정 (감속 + 가속 보너스)
      8. RViz 마커 퍼블리시
    """

    def __init__(self):
        super().__init__('gap_follow')

        # =========================
        # Parameter declaration
        # =========================
        self.declare_parameter('control_rate_hz', 50.0)

        # -------------------------
        # CORE GAP FOLLOW PARAMS
        # -------------------------
        self.declare_parameter('gf_speed', 0.5)
        self.declare_parameter('gf_min_speed', 0.1)
        self.declare_parameter('gf_max_steer', 0.4)
        self.declare_parameter('gf_max_range', 10.0)
        self.declare_parameter('gf_fov_deg', 140.0)
        self.declare_parameter('gf_use_bubble', False)
        self.declare_parameter('gf_bubble_radius', 0.4)
        self.declare_parameter('gf_gap_distance_threshold', 1.0)
        self.declare_parameter('gf_gap_half_beam_count', 0)
        self.declare_parameter('gf_gap_use_dynamic_center', True)
        self.declare_parameter('gf_gap_center_gain', 1.0)
        self.declare_parameter('gf_gap_max_center_shift_deg', 35.0)
        self.declare_parameter('gf_corner_opposite_beam_scale', 0.45)
        self.declare_parameter('gf_turn_hint_center_deg', 12.0)
        self.declare_parameter('gf_turn_hint_side_start_deg', 20.0)
        self.declare_parameter('gf_turn_hint_side_end_deg', 75.0)
        self.declare_parameter('gf_turn_open_ratio', 1.15)
        self.declare_parameter('gf_turn_diff_min', 0.30)
        self.declare_parameter('gf_turn_confidence_gain', 1.35)
        self.declare_parameter('gf_corner_center_shift_deg', 18.0)
        self.declare_parameter('gf_corner_target_bias', 0.72)
        self.declare_parameter('gf_gap_target_depth_bias', 0.75)
        self.declare_parameter('gf_left_corner_center_shift_gain', 1.0)
        self.declare_parameter('gf_left_corner_target_depth_gain', 1.0)
        self.declare_parameter('gf_left_corner_min_steer_ratio', 0.0)
        self.declare_parameter('gf_brake_distance', 5.0)
        self.declare_parameter('gf_turn_max_range_reduction', 0.0)
        self.declare_parameter('gf_turn_speed_drop_gain', 1.0)
        self.declare_parameter('gf_left_hook_open_delta', 0.0)
        self.declare_parameter('gf_left_hook_open_distance', 0.0)
        self.declare_parameter('gf_left_hook_steer_ratio', 0.0)
        self.declare_parameter('gf_front_speed_check_angle_deg', 20.0)
        self.declare_parameter('gf_front_safe_speed_up_distance', 0.0)
        self.declare_parameter('gf_front_safe_speed_up_gain', 0.0)
        self.declare_parameter('gf_kp', 1.0)
        self.declare_parameter('gf_ki', 0.0)
        self.declare_parameter('gf_kd', 0.0)
        self.declare_parameter('gf_target_speed_boost_start_ratio', 0.75)
        self.declare_parameter('gf_target_speed_boost_gain', 0.60)
        self.declare_parameter('viz_frame', 'base_link')
        self.declare_parameter('viz_gap_topic', '/gf/gap_marker')
        self.declare_parameter('viz_bubble_topic', '/gf/bubble_marker')
        self.declare_parameter('viz_gap_center_topic', '/gf/gap_center_marker')
        self.declare_parameter('viz_target_point_topic', '/gf/target_point_marker')
        self.declare_parameter('viz_target_line_topic', '/gf/target_line_marker')

        p = lambda name: self.get_parameter(name).value

        # =========================
        # Load params
        # =========================
        self.control_rate_hz = float(p('control_rate_hz'))

        # -- 기본 속도 / 조향 --
        self.gf_speed = float(p('gf_speed'))         # 직선 최고 속도 [m/s]
        self.gf_min_speed = float(p('gf_min_speed')) # 코너 최저 속도 [m/s]
        self.max_steer = float(p('gf_max_steer'))    # 최대 조향각 [rad]

        # -- LiDAR 탐색 범위 --
        self.max_range = float(p('gf_max_range'))    # Gap 탐색 최대 거리 [m]
        self.fov_deg = float(p('gf_fov_deg'))        # 유효 시야각 [deg]

        # -- Safety Bubble --
        self.use_bubble = bool(p('gf_use_bubble'))         # Bubble 마스킹 활성화 여부
        self.bubble_radius = float(p('gf_bubble_radius'))  # Bubble 반경 [m]

        # -- Gap 탐지 기준 --
        self.gap_distance_threshold = float(p('gf_gap_distance_threshold'))   # Gap 인식 최소 거리 [m]
        self.gap_half_beam_count = max(0, int(p('gf_gap_half_beam_count')))   # Gap 탐색 빔 수 (0=전체 FOV)

        # -- 동적 Gap 중심 이동 --
        self.gap_use_dynamic_center = bool(p('gf_gap_use_dynamic_center'))         # 조향 방향으로 탐색 중심 이동 여부
        self.gap_center_gain = float(p('gf_gap_center_gain'))                       # 이전 조향 × gain → 중심 이동량
        self.gap_max_center_shift_deg = float(p('gf_gap_max_center_shift_deg'))    # 중심 이동 최대 각도 [deg]

        # -- 커브 진입 시 반대 방향 빔 차단 --
        self.corner_opposite_beam_scale = float(p('gf_corner_opposite_beam_scale'))

        # -- 커브 감지: 정면 / 측면 openness 비교 섹터 범위 --
        self.turn_hint_center_deg = float(p('gf_turn_hint_center_deg'))           # 정면 섹터 반폭 [deg]
        self.turn_hint_side_start_deg = float(p('gf_turn_hint_side_start_deg'))   # 측면 섹터 시작 각도 [deg]
        self.turn_hint_side_end_deg = float(p('gf_turn_hint_side_end_deg'))       # 측면 섹터 끝 각도 [deg]
        self.turn_open_ratio = float(p('gf_turn_open_ratio'))                     # 커브 후보 판정 비율 (측면/정면)
        self.turn_diff_min = float(p('gf_turn_diff_min'))                         # 좌/우 최소 openness 차이 [m]
        self.turn_confidence_gain = float(p('gf_turn_confidence_gain'))           # 커브 확신도 누적 속도

        # -- 커브 확정 시 Gap 탐색 조정 --
        self.corner_center_shift_deg = float(p('gf_corner_center_shift_deg'))   # 탐색 중심 추가 이동 각도 [deg]
        self.corner_target_bias = float(p('gf_corner_target_bias'))             # 목표점 Gap 내 위치 (0.5=중앙, 1.0=끝)

        # -- 목표 지점 선택: 깊이(depth) vs 중앙(center) 가중치 --
        self.gap_target_depth_bias = float(p('gf_gap_target_depth_bias'))

        # -- 좌코너 전용 보정 계수 --
        self.left_corner_center_shift_gain = float(p('gf_left_corner_center_shift_gain'))   # 좌코너 중심 이동 강도
        self.left_corner_target_depth_gain = float(p('gf_left_corner_target_depth_gain'))   # 좌코너 깊이 목표 강도
        self.left_corner_min_steer_ratio = float(p('gf_left_corner_min_steer_ratio'))       # 좌코너 최소 조향 비율

        # -- 감속 --
        self.brake_distance = float(p('gf_brake_distance'))             # 전방 감속 시작 거리 [m]
        self.turn_max_range_reduction = float(p('gf_turn_max_range_reduction'))  # 커브 시 최대 탐색 범위 축소량 [m]
        self.turn_speed_drop_gain = float(p('gf_turn_speed_drop_gain'))          # 코너링 속도 감소 강도

        # -- 갑작스러운 전방 개방 감지 (Left Hook 대응) --
        self.left_hook_open_delta = float(p('gf_left_hook_open_delta'))       # 전방 거리 급증 감지 임계값 [m]
        self.left_hook_open_distance = float(p('gf_left_hook_open_distance')) # 전방 개방 판정 거리 [m]
        self.left_hook_steer_ratio = float(p('gf_left_hook_steer_ratio'))     # Left Hook 시 강제 조향 비율

        # -- 정면 개방도 기반 가속 --
        self.front_speed_check_angle_deg = float(p('gf_front_speed_check_angle_deg'))       # 정면 거리 계산 각도 폭 [deg]
        self.front_safe_speed_up_distance = float(p('gf_front_safe_speed_up_distance'))     # 가속 허용 최소 정면 거리 [m]
        self.front_safe_speed_up_gain = float(p('gf_front_safe_speed_up_gain'))             # 정면 개방 가속 강도
        
        # -- PID 조향 제어 계수 --
        self.gf_kp = float(p('gf_kp'))   # 비례(P): 오차에 즉각 반응
        self.gf_ki = float(p('gf_ki'))   # 적분(I): 누적 오차 보정
        self.gf_kd = float(p('gf_kd'))   # 미분(D): 오차 변화율 억제

        # -- 목표 Gap 거리 기반 직선 가속 --
        self.target_speed_boost_start_ratio = float(p('gf_target_speed_boost_start_ratio'))  # 가속 시작 거리 비율
        self.target_speed_boost_gain = float(p('gf_target_speed_boost_gain'))                # 가속 강도

        self.viz_frame = str(p('viz_frame'))

        # =========================
        # State (제어 루프 간 유지되는 상태값)
        # =========================
        self.scan = None               # 최신 LiDAR 스캔 데이터
        self.odom = None               # 최신 오도메트리 데이터
        self.prev_error = 0.0          # PID 미분 계산용 이전 오차
        self.integral_error = 0.0      # PID 적분 누적값
        self.prev_steer_cmd = 0.0      # 동적 Gap 중심 이동에 사용할 이전 조향 명령 [rad]
        self.prev_front_dist = 0.0     # Left Hook 감지용 이전 프레임 전방 거리 [m]


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

        self.gap_viz_pub = self.create_publisher(Marker, str(p('viz_gap_topic')), 10)
        self.bubble_viz_pub = self.create_publisher(Marker, str(p('viz_bubble_topic')), 10)
        self.gap_center_viz_pub = self.create_publisher(Marker, str(p('viz_gap_center_topic')), 10)
        self.target_point_viz_pub = self.create_publisher(Marker, str(p('viz_target_point_topic')), 10)
        self.target_line_viz_pub = self.create_publisher(Marker, str(p('viz_target_line_topic')), 10)

        self.create_timer(1.0 / self.control_rate_hz, self._loop)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _loop(self):
        """제어 루프: scan/odom이 모두 수신된 경우에만 실행."""
        if self.scan is None or self.odom is None:
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        """
        메인 알고리즘: 스캔 데이터를 처리해 조향각과 속도를 반환.

        Returns:
            steer (float): 조향 명령 [rad], + = 좌회전
            speed (float): 속도 명령 [m/s]
        """
        ranges = np.array(self.scan.ranges, dtype=np.float32)
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        n = len(ranges)
        beam_angles = angle_min + np.arange(n, dtype=np.float32) * angle_inc

        # -------------------------------------------------------
        # Step 1. 스캔 전처리: NaN/Inf 제거 및 최대 거리 클리핑
        # -------------------------------------------------------
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)

        # -------------------------------------------------------
        # Step 2. 전방 평균 거리 계산 → 사전 감속 속도 허용치 결정
        #
        # 정면 ±(front_speed_check_angle_deg/2) 범위의 빔 평균으로
        # front_dist를 구하고, brake_distance와 비교해 허용 최고 속도를 선형으로 제한.
        # (front_dist가 작을수록 allowed_max_speed가 gf_min_speed에 가까워짐)
        # -------------------------------------------------------
        front_angle_width = math.radians(max(self.front_speed_check_angle_deg, 1.0))
        front_start_idx = max(0, int((-front_angle_width / 2.0 - angle_min) / angle_inc))
        front_end_idx = min(n - 1, int((front_angle_width / 2.0 - angle_min) / angle_inc))
        front_ranges = ranges[front_start_idx:front_end_idx + 1]
        front_dist = float(np.mean(front_ranges)) if len(front_ranges) > 0 else self.max_range
        prev_front_dist = self.prev_front_dist
        self.prev_front_dist = front_dist
        
        # 전방 거리 비율로 이번 프레임의 허용 최고 속도 계산
        allowed_max_speed = self.gf_min_speed + (self.gf_speed - self.gf_min_speed) * min(1.0, front_dist / self.brake_distance)

        # -------------------------------------------------------
        # Step 3. FOV 마스킹: 설정된 시야각 바깥 빔은 0으로 처리
        # -------------------------------------------------------
        fov_rad = math.radians(self.fov_deg)
        min_angle = -fov_rad / 2.0
        max_angle = fov_rad / 2.0
        
        start_idx = max(0, int((min_angle - angle_min) / angle_inc))
        end_idx = min(n - 1, int((max_angle - angle_min) / angle_inc))
        
        fov_ranges = np.zeros_like(ranges)
        fov_ranges[start_idx:end_idx + 1] = ranges[start_idx:end_idx + 1]

        # 커브 감지 및 커브 확신도에 따른 유효 탐색 거리 계산
        turn_hint, turn_confidence = self._detect_turn_hint(fov_ranges, beam_angles)
        effective_max_range = self._compute_effective_max_range(turn_hint, turn_confidence)

        # 유효 탐색 거리로 클리핑한 처리용 배열 생성
        eval_ranges = np.minimum(ranges, effective_max_range)
        proc_ranges = np.zeros_like(eval_ranges)
        proc_ranges[start_idx:end_idx + 1] = eval_ranges[start_idx:end_idx + 1]

        # -------------------------------------------------------
        # Step 4. 가장 가까운 장애물 탐색
        # -------------------------------------------------------
        valid_indices = np.where(proc_ranges > 0.0)[0]
        if len(valid_indices) == 0:
            self._clear_debug_markers()
            return 0.0, self.gf_min_speed

        nearest_idx = valid_indices[np.argmin(proc_ranges[valid_indices])]
        nearest_dist = proc_ranges[nearest_idx]
        bubble_bounds = None

        # -------------------------------------------------------
        # Step 5. Safety Bubble 적용 (use_bubble=True 일 때)
        #
        # 가장 가까운 장애물을 중심으로 bubble_radius 각도 범위를
        # 0으로 마스킹해 차량이 해당 방향으로 Gap을 선택하지 않게 함.
        # -------------------------------------------------------
        if self.use_bubble and nearest_dist > 0.0:
            bubble_angle = math.asin(min(self.bubble_radius / nearest_dist, 1.0))
            bubble_idx_radius = int(bubble_angle / angle_inc)
            
            b_start = max(0, nearest_idx - bubble_idx_radius)
            b_end = min(n - 1, nearest_idx + bubble_idx_radius)
            proc_ranges[b_start:b_end + 1] = 0.0
            bubble_bounds = (b_start, b_end)

        # -------------------------------------------------------
        # Step 6. Gap 탐색 범위(윈도우) 설정
        #
        # - gap_use_dynamic_center: 이전 조향각 방향으로 탐색 중심 이동
        # - 커브 감지 시: 커브 방향으로 추가 이동 + 반대쪽 빔 축소
        # - gap_half_beam_count > 0: 중심에서 ±N 빔만 탐색
        # -------------------------------------------------------
        gap_candidate_ranges = np.zeros_like(proc_ranges)
        gap_eval_start = start_idx
        gap_eval_end = end_idx
        gap_center_angle = 0.0

        # 이전 조향 방향으로 탐색 중심 이동 (관성 유지)
        if self.gap_use_dynamic_center:
            gap_center_angle = self.gap_center_gain * self.prev_steer_cmd

        # 커브 확신 시 탐색 중심을 커브 방향으로 추가 이동
        if turn_hint != 0 and turn_confidence > 0.0:
            corner_shift_gain = self.left_corner_center_shift_gain if turn_hint > 0 else 1.0
            gap_center_angle += (
                math.radians(self.corner_center_shift_deg)
                * turn_hint
                * turn_confidence
                * corner_shift_gain
            )

        max_shift = math.radians(self.gap_max_center_shift_deg)
        gap_center_angle = float(np.clip(gap_center_angle, -max_shift, max_shift))

        # 반고정 빔 수 방식: 중심에서 좌/우 N빔만 탐색 (커브 시 반대 방향 빔 축소)
        if self.gap_half_beam_count > 0:
            center_idx = int(np.argmin(np.abs(beam_angles - gap_center_angle)))
            left_beam_count = self.gap_half_beam_count
            right_beam_count = self.gap_half_beam_count
            if turn_hint != 0 and turn_confidence > 0.0:
                opposite_scale = float(np.clip(self.corner_opposite_beam_scale, 0.05, 1.0))
                blended_scale = 1.0 - (1.0 - opposite_scale) * turn_confidence
                reduced_count = max(1, int(round(self.gap_half_beam_count * blended_scale)))
                if turn_hint > 0:   # 좌커브: 오른쪽 빔 축소
                    right_beam_count = reduced_count
                else:               # 우커브: 왼쪽 빔 축소
                    left_beam_count = reduced_count
            gap_eval_start = max(start_idx, center_idx - left_beam_count)
            gap_eval_end = min(end_idx, center_idx + right_beam_count)

        gap_candidate_ranges[gap_eval_start:gap_eval_end + 1] = proc_ranges[gap_eval_start:gap_eval_end + 1]

        # gap_distance_threshold 미만인 빔은 Gap 후보에서 제외
        effective_gap_distance_threshold = min(self.gap_distance_threshold, effective_max_range)
        gap_candidate_ranges[gap_candidate_ranges < effective_gap_distance_threshold] = 0.0

        valid_gaps = np.where(gap_candidate_ranges > 0.0)[0]
        if len(valid_gaps) == 0:
            # 유효 Gap 없음 → 최소 속도 직진
            self._publish_debug_markers(
                ranges=eval_ranges,
                angle_min=angle_min,
                angle_inc=angle_inc,
                gap_start=None,
                gap_end=None,
                best_idx=None,
                nearest_idx=nearest_idx,
                nearest_dist=nearest_dist,
                bubble_bounds=bubble_bounds,
            )
            return 0.0, self.gf_min_speed

        # 연속된 빔 묶음(Gap 후보군)으로 분리 후 가장 큰 Gap 선택
        splits = np.split(valid_gaps, np.where(np.diff(valid_gaps) > 1)[0] + 1)
        max_gap = self._select_largest_gap(splits, gap_candidate_ranges)
        gap_start, gap_end = int(max_gap[0]), int(max_gap[-1])

        # -------------------------------------------------------
        # Step 7. Gap 내 목표 지점 선택
        #
        # gap_center_idx (Gap 중앙)와 deepest_idx (Gap 내 가장 먼 지점)를
        # target_depth_bias 비율로 보간.
        # - 장애물이 정면에 가까울수록 + 커브 확신도가 높을수록 deepest_idx 비중 증가
        # -------------------------------------------------------
        gap_center_idx = (gap_start + gap_end) // 2
        deepest_idx = gap_start + int(np.argmax(gap_candidate_ranges[gap_start:gap_end + 1]))

        # 장애물이 정면 중앙에 얼마나 가까이 있는지 (0~1)
        center_half = math.radians(max(self.turn_hint_center_deg, 1.0))
        obstacle_centered_confidence = float(
            np.clip(1.0 - abs(float(beam_angles[nearest_idx])) / max(center_half, 1e-3), 0.0, 1.0)
        )
        # 장애물이 얼마나 가까이 있는지 (0~1)
        obstacle_close_confidence = float(
            np.clip(1.0 - nearest_dist / max(effective_max_range, 1e-3), 0.0, 1.0)
        )
        # 두 요소를 곱해 "정면 근거리 장애물" 확신도 계산
        obstacle_ahead_confidence = obstacle_centered_confidence * obstacle_close_confidence

        # 커브 확신도와 장애물 확신도 중 큰 값으로 목표 지점 선택에 활용
        target_selection_confidence = max(turn_confidence, obstacle_ahead_confidence)
        target_depth_bias_gain = self.left_corner_target_depth_gain if turn_hint > 0 else 1.0
        target_depth_bias = (
            float(np.clip(self.gap_target_depth_bias, 0.0, 1.0))
            * target_selection_confidence
            * target_depth_bias_gain
        )
        target_depth_bias = float(np.clip(target_depth_bias, 0.0, 1.0))

        # gap_center_idx와 deepest_idx 사이를 target_depth_bias 비율로 보간
        best_idx = int(round((1.0 - target_depth_bias) * gap_center_idx + target_depth_bias * deepest_idx))
        best_idx = int(np.clip(best_idx, gap_start, gap_end))
        best_angle = angle_min + best_idx * angle_inc
        target_dist = float(eval_ranges[best_idx])
        
        # -------------------------------------------------------
        # Step 8. PID 조향 제어
        #
        # 오차 = 목표 지점의 각도 (0도 정면 기준)
        # -------------------------------------------------------
        error = best_angle  # 0도(정면)를 기준으로 목표 지점의 각도를 오차로 간주
        
        # 적분 누적(Anti-windup): I값이 너무 커져서 통제 불능이 되는 것 방지
        self.integral_error += error
        self.integral_error = float(np.clip(self.integral_error, -2.0, 2.0))
        
        # 미분 오차 계산 (현재 오차 - 이전 오차)
        derivative = error - self.prev_error
        self.prev_error = error
        
        # PID 계산식 적용
        pid_steer = (self.gf_kp * error) + (self.gf_ki * self.integral_error) + (self.gf_kd * derivative)
        
        raw_steer = float(np.clip(pid_steer, -self.max_steer, self.max_steer))
        
        # -------------------------------------------------------
        # Step 9. 조향 보정
        #
        # (A) 직선 구간 조향 억제: 전방이 충분히 열려있고 커브가 아닐 때
        #     조향 입력을 최대 90% 감쇄 → 미세 진동 방지
        # (B) Left Hook 대응: 전방이 갑자기 열릴 때 강제 좌향 조향
        # (C) 좌코너 최소 조향 보장: 좌커브 확신 시 최소 조향각 하한 설정
        # -------------------------------------------------------
        straight_confidence = self._range_confidence(front_dist, self.front_safe_speed_up_distance, self.max_range)
        straight_confidence *= (1.0 - turn_confidence)  # 커브 중에는 직선 억제 해제
        sudden_open_confidence = self._compute_left_hook_confidence(front_dist, prev_front_dist)

        # (A) 직선 조향 억제
        raw_steer *= (1.0 - 0.90 * straight_confidence)

        # (B) Left Hook 강제 조향
        if sudden_open_confidence > 0.0:
            forced_left_steer = self.max_steer * float(
                np.clip(self.left_hook_steer_ratio * sudden_open_confidence, 0.0, 1.0)
            )
            raw_steer = max(raw_steer, forced_left_steer)

        # (C) 좌코너 최소 조향
        if turn_hint > 0 and turn_confidence > 0.0:
            left_corner_min_steer = self.max_steer * float(
                np.clip(self.left_corner_min_steer_ratio * turn_confidence, 0.0, 1.0)
            )
            raw_steer = max(raw_steer, left_corner_min_steer)

        target_range_ratio = float(np.clip(target_dist / max(effective_max_range, 1e-3), 0.0, 1.0))

        # 3.5도 미만의 얕은 조향은 직진으로 강제 (미세 진동 완전 차단)
        if abs(raw_steer) < math.radians(3.5):
            raw_steer = 0.0

        steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))
        self.prev_steer_cmd = steer
        
        steer_ratio = abs(steer) / self.max_steer if self.max_steer > 1e-6 else 0.0
        
        # -------------------------------------------------------
        # Step 10. 속도 결정
        #
        # 기준: allowed_max_speed (Step 2의 사전 감속 상한)
        # 1) 코너링 감속: steer_ratio^3 으로 조향 비율에 따라 속도 감소
        #    (3제곱: 조향각이 작을 땐 속도 거의 안 깎고, 클 때만 크게 감소)
        # 2) 목표 Gap 거리 가속 보너스
        # 3) 정면 개방도 가속 보너스
        # -------------------------------------------------------
        speed_drop_ratio = float(
            np.clip(self.turn_speed_drop_gain, 0.0, 1.0) * (steer_ratio ** 3.0)
        )
        speed = float(allowed_max_speed - speed_drop_ratio * (allowed_max_speed - self.gf_min_speed))

        # 가속 보너스 1: 목표 Gap이 멀리 있고, 커브/조향 없을 때
        boost_start = float(np.clip(self.target_speed_boost_start_ratio, 0.0, 0.99))
        target_open_confidence = float(np.clip((target_range_ratio - boost_start) / max(1.0 - boost_start, 1e-3), 0.0, 1.0))
        target_open_confidence *= (1.0 - turn_confidence)  # 커브 중 보너스 억제
        target_open_confidence *= (1.0 - steer_ratio)      # 조향 중 보너스 억제
        boost_gain = float(np.clip(self.target_speed_boost_gain, 0.0, 1.5))
        speed += boost_gain * target_open_confidence * (self.gf_speed - speed)

        # 가속 보너스 2: 정면이 충분히 열려 있을 때
        front_open_confidence = self._range_confidence(front_dist, self.front_safe_speed_up_distance, self.max_range)
        front_open_confidence *= (1.0 - turn_confidence)
        front_open_confidence *= (1.0 - 0.5 * steer_ratio)
        front_boost_gain = float(np.clip(self.front_safe_speed_up_gain, 0.0, 1.5))
        speed += front_boost_gain * front_open_confidence * (self.gf_speed - speed)

        speed = float(np.clip(speed, self.gf_min_speed, self.gf_speed))

        self._publish_debug_markers(
            ranges=eval_ranges,
            angle_min=angle_min,
            angle_inc=angle_inc,
            gap_start=gap_start,
            gap_end=gap_end,
            best_idx=best_idx,
            nearest_idx=nearest_idx,
            nearest_dist=nearest_dist,
            bubble_bounds=bubble_bounds,
        )

        return steer, speed

    # =========================================================
    # RViz 마커 유틸리티
    # =========================================================

    def _make_marker(self, marker_id, ns, marker_type):
        """기본 마커 객체 생성 (공통 헤더/포즈 세팅)."""
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _delete_marker(self, pub, marker_id, ns, marker_type):
        """지정 마커를 DELETE 액션으로 퍼블리시해 RViz에서 제거."""
        marker = self._make_marker(marker_id, ns, marker_type)
        marker.action = Marker.DELETE
        pub.publish(marker)

    def _clear_debug_markers(self):
        """유효 Gap이 없을 때 모든 디버그 마커를 RViz에서 제거."""
        self._delete_marker(self.gap_viz_pub, 0, 'gf_gap', Marker.LINE_STRIP)
        self._delete_marker(self.bubble_viz_pub, 0, 'gf_bubble', Marker.SPHERE)
        self._delete_marker(self.gap_center_viz_pub, 0, 'gf_gap_center', Marker.SPHERE)
        self._delete_marker(self.target_point_viz_pub, 0, 'gf_target_point', Marker.SPHERE)
        self._delete_marker(self.target_line_viz_pub, 0, 'gf_target_line', Marker.LINE_STRIP)

    def _publish_debug_markers(
        self,
        ranges,
        angle_min,
        angle_inc,
        gap_start,
        gap_end,
        best_idx,
        nearest_idx,
        nearest_dist,
        bubble_bounds,
    ):
        """모든 디버그 마커를 한 번에 퍼블리시."""
        self._publish_gap_marker(ranges, angle_min, angle_inc, gap_start, gap_end)
        self._publish_bubble_marker(ranges, angle_min, angle_inc, nearest_idx, nearest_dist, bubble_bounds)
        self._publish_gap_center_marker(ranges, angle_min, angle_inc, gap_start, gap_end)
        self._publish_target_point_marker(ranges, angle_min, angle_inc, best_idx)
        self._publish_target_line_marker(ranges, angle_min, angle_inc, best_idx)

    # =========================================================
    # 알고리즘 헬퍼 함수들
    # =========================================================

    def _select_largest_gap(self, gaps, gap_candidate_ranges):
        """
        Gap 후보 목록에서 '가장 큰' Gap을 선택.

        점수 기준: (빔 수(너비), 평균 거리(깊이)) 의 튜플로 비교.
        너비가 같으면 더 깊은(먼) Gap을 선택.
        """
        def gap_score(gap):
            gap = np.asarray(gap)
            start = int(gap[0])
            end = int(gap[-1])
            width = end - start + 1
            mean_depth = float(np.mean(gap_candidate_ranges[start:end + 1]))
            return (width, mean_depth)

        return max(gaps, key=gap_score)

    def _compute_effective_max_range(self, turn_hint, turn_confidence):
        """
        커브 확신도에 비례해 Gap 탐색 최대 거리를 줄임.

        커브 중에 먼 곳의 허위 Gap을 무시하고 가까운 실제 경로에 집중하기 위함.
        """
        reduction = max(0.0, self.turn_max_range_reduction)
        if turn_hint == 0 or turn_confidence <= 0.0 or reduction <= 0.0:
            return self.max_range
        min_range = max(0.3, 0.1 * self.max_range)  # 최소 탐색 거리 하한 보장
        reduced_range = self.max_range - reduction * float(np.clip(turn_confidence, 0.0, 1.0))
        return float(np.clip(reduced_range, min_range, self.max_range))

    def _range_confidence(self, distance, start_distance, max_distance):
        """
        거리를 [start_distance, max_distance] 구간에서 [0, 1]로 정규화.

        distance < start_distance → 0.0 (아직 충분히 열리지 않음)
        distance >= max_distance  → 1.0 (완전히 열림)
        """
        start = float(np.clip(start_distance, 0.0, max_distance))
        if max_distance <= start + 1e-3:
            return 0.0
        return float(np.clip((distance - start) / (max_distance - start), 0.0, 1.0))

    def _compute_left_hook_confidence(self, front_dist, prev_front_dist):
        """
        전방 거리가 갑자기 크게 증가했을 때 Left Hook 가능성 계산.

        조건:
          - 이번 프레임과 이전 프레임 간 전방 거리 증가분 > left_hook_open_delta
          - 현재 전방 거리 > left_hook_open_distance
          - left_hook_steer_ratio > 0 (기능 활성화)

        Returns:
            confidence (float): 0.0 ~ 1.0
        """
        delta_threshold = max(0.0, self.left_hook_open_delta)
        open_distance = max(0.0, self.left_hook_open_distance)
        steer_ratio = max(0.0, self.left_hook_steer_ratio)
        if delta_threshold <= 0.0 or steer_ratio <= 0.0:
            return 0.0
        front_open_delta = max(0.0, front_dist - prev_front_dist)
        delta_confidence = self._range_confidence(front_open_delta, delta_threshold, self.max_range)
        distance_confidence = self._range_confidence(front_dist, open_distance, self.max_range)
        return float(delta_confidence * distance_confidence)

    def _detect_turn_hint(self, proc_ranges, beam_angles):
        """
        LiDAR 데이터로 커브 방향과 확신도를 추정.

        정면 섹터와 좌/우 측면 섹터의 70퍼센타일 거리를 비교:
          - 좌측이 정면보다 turn_open_ratio 이상 열려 있으면 → 좌커브 (hint=+1)
          - 우측이 정면보다 turn_open_ratio 이상 열려 있으면 → 우커브 (hint=-1)
          - 해당 없음 → (0, 0.0)

        Returns:
            turn_hint (int): +1(좌), -1(우), 0(직선)
            turn_confidence (float): 0.0 ~ 1.0
        """
        center_half = math.radians(self.turn_hint_center_deg)
        side_start = math.radians(self.turn_hint_side_start_deg)
        side_end = math.radians(self.turn_hint_side_end_deg)

        center_open = self._sector_percentile(proc_ranges, beam_angles, -center_half, center_half)
        left_open = self._sector_percentile(proc_ranges, beam_angles, side_start, side_end)
        right_open = self._sector_percentile(proc_ranges, beam_angles, -side_end, -side_start)

        # 정면이 완전히 막혀 있으면 좌/우 중 큰 값으로 대체 (막힌 코너 대응)
        if center_open <= 0.0:
            center_open = min(max(left_open, right_open), self.max_range)

        left_advantage = left_open - right_open
        right_advantage = right_open - left_open

        if left_open > center_open * self.turn_open_ratio and left_advantage > self.turn_diff_min:
            confidence = self._turn_confidence(left_advantage, center_open, left_open)
            return 1, confidence
        if right_open > center_open * self.turn_open_ratio and right_advantage > self.turn_diff_min:
            confidence = self._turn_confidence(right_advantage, center_open, right_open)
            return -1, confidence
        return 0, 0.0

    def _turn_confidence(self, side_advantage, center_open, side_open):
        """
        커브 확신도 계산.

        두 항 중 큰 값을 기반으로 [0, 1] 범위로 정규화 후 gain 적용:
          - diff_term: side_advantage / max_range (절대적 차이 비율)
          - ratio_term: side_open / center_open - 1.0 (상대적 개방 비율)
        """
        diff_term = float(np.clip(side_advantage / max(self.max_range, 1e-3), 0.0, 1.0))
        ratio_term = float(np.clip((side_open / max(center_open, 1e-3)) - 1.0, 0.0, 1.0))
        base_confidence = max(diff_term, ratio_term)
        return float(np.clip(base_confidence * self.turn_confidence_gain, 0.0, 1.0))

    def _sector_percentile(self, ranges, angles, start_angle, end_angle, percentile=70.0):
        """
        지정 각도 섹터 내 유효 빔(ranges > 0)의 percentile 거리 반환.

        평균 대신 70퍼센타일 사용: 이상치(매우 가까운 물체)에 덜 민감.
        유효 빔 없으면 0.0 반환.
        """
        low = min(start_angle, end_angle)
        high = max(start_angle, end_angle)
        mask = (angles >= low) & (angles <= high) & (ranges > 0.0)
        if not np.any(mask):
            return 0.0
        return float(np.percentile(ranges[mask], percentile))

    # =========================================================
    # RViz 마커 퍼블리시 함수들
    # =========================================================

    def _publish_gap_marker(self, ranges, angle_min, angle_inc, gap_start, gap_end):
        """선택된 Gap 범위를 청록색 LINE_STRIP으로 표시."""
        if gap_start is None or gap_end is None or gap_end < gap_start:
            self._delete_marker(self.gap_viz_pub, 0, 'gf_gap', Marker.LINE_STRIP)
            return

        marker = self._make_marker(0, 'gf_gap', Marker.LINE_STRIP)
        marker.scale.x = 0.05
        marker.color = ColorRGBA(r=0.1, g=0.8, b=1.0, a=0.95)

        for idx in range(gap_start, gap_end + 1):
            dist = float(ranges[idx])
            if dist <= 0.0:
                continue
            angle = angle_min + idx * angle_inc
            marker.points.append(
                Point(
                    x=float(dist * math.cos(angle)),
                    y=float(dist * math.sin(angle)),
                    z=0.0,
                )
            )

        if len(marker.points) < 2:
            self._delete_marker(self.gap_viz_pub, 0, 'gf_gap', Marker.LINE_STRIP)
            return

        self.gap_viz_pub.publish(marker)

    def _publish_bubble_marker(self, ranges, angle_min, angle_inc, nearest_idx, nearest_dist, bubble_bounds):
        """Safety Bubble을 가장 가까운 장애물 위치에 빨간 반투명 구로 표시."""
        if (not self.use_bubble) or nearest_idx is None or nearest_dist is None or nearest_dist <= 0.0 or bubble_bounds is None:
            self._delete_marker(self.bubble_viz_pub, 0, 'gf_bubble', Marker.SPHERE)
            return

        nearest_angle = angle_min + nearest_idx * angle_inc
        marker = self._make_marker(0, 'gf_bubble', Marker.SPHERE)
        marker.pose.position.x = float(nearest_dist * math.cos(nearest_angle))
        marker.pose.position.y = float(nearest_dist * math.sin(nearest_angle))
        marker.pose.position.z = 0.0
        marker.scale.x = float(self.bubble_radius * 2.0)
        marker.scale.y = float(self.bubble_radius * 2.0)
        marker.scale.z = 0.05
        marker.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.28)
        self.bubble_viz_pub.publish(marker)

    def _publish_gap_center_marker(self, ranges, angle_min, angle_inc, gap_start, gap_end):
        """Gap 중앙 인덱스 위치를 주황색 구로 표시."""
        if gap_start is None or gap_end is None or gap_end < gap_start:
            self._delete_marker(self.gap_center_viz_pub, 0, 'gf_gap_center', Marker.SPHERE)
            return

        gap_center_idx = (gap_start + gap_end) // 2
        dist = float(ranges[gap_center_idx])
        if dist <= 0.0:
            self._delete_marker(self.gap_center_viz_pub, 0, 'gf_gap_center', Marker.SPHERE)
            return

        angle = angle_min + gap_center_idx * angle_inc
        marker = self._make_marker(0, 'gf_gap_center', Marker.SPHERE)
        marker.pose.position.x = float(dist * math.cos(angle))
        marker.pose.position.y = float(dist * math.sin(angle))
        marker.pose.position.z = 0.0
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22
        marker.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
        self.gap_center_viz_pub.publish(marker)

    def _publish_target_point_marker(self, ranges, angle_min, angle_inc, best_idx):
        """최종 목표 지점을 초록색 구로 표시."""
        if best_idx is None:
            self._delete_marker(self.target_point_viz_pub, 0, 'gf_target_point', Marker.SPHERE)
            return

        dist = float(ranges[best_idx])
        if dist <= 0.0:
            self._delete_marker(self.target_point_viz_pub, 0, 'gf_target_point', Marker.SPHERE)
            return

        angle = angle_min + best_idx * angle_inc
        marker = self._make_marker(0, 'gf_target_point', Marker.SPHERE)
        marker.pose.position.x = float(dist * math.cos(angle))
        marker.pose.position.y = float(dist * math.sin(angle))
        marker.pose.position.z = 0.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)
        self.target_point_viz_pub.publish(marker)

    def _publish_target_line_marker(self, ranges, angle_min, angle_inc, best_idx):
        """차량 원점에서 목표 지점까지 노란색 선으로 표시."""
        if best_idx is None:
            self._delete_marker(self.target_line_viz_pub, 0, 'gf_target_line', Marker.LINE_STRIP)
            return

        dist = float(ranges[best_idx])
        if dist <= 0.0:
            self._delete_marker(self.target_line_viz_pub, 0, 'gf_target_line', Marker.LINE_STRIP)
            return

        angle = angle_min + best_idx * angle_inc
        marker = self._make_marker(0, 'gf_target_line', Marker.LINE_STRIP)
        marker.scale.x = 0.035
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.1, a=0.95)
        marker.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(
                x=float(dist * math.cos(angle)),
                y=float(dist * math.sin(angle)),
                z=0.0,
            ),
        ]
        self.target_line_viz_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 종료 시 차량 정지 명령 전송
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