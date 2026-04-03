import math
import numpy as np
import rclpy
from rclpy.node import Node

# 라이다 데이터 메시지
from sensor_msgs.msg import LaserScan

# 오도메트리 메시지
from nav_msgs.msg import Odometry

# Ackermann 조향/속도 명령 메시지
from ackermann_msgs.msg import AckermannDriveStamped

# RViz 시각화를 위한 Marker 메시지
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

# 우리가 만든 E-Stop 클래스
from controller.estop import EStop


class GapFollowNode(Node):
    # Gap Following 기반 주행 노드
    # 주요 역할:
    # 1) 라이다로 전방 free space 탐색
    # 2) 가장 안전한 gap 찾기
    # 3) 커브 힌트와 한쪽 벽 prior를 반영
    # 4) 최종 steering/speed 계산
    # 5) 마지막에 EStop으로 안전 검사
    # 6) RViz 디버그 마커 publish

    def __init__(self):
        super().__init__('gap_follow')
        # ROS2 노드 이름을 'gap_follow'로 생성

        # =========================
        # Parameter declaration
        # =========================
        self.declare_parameter('control_rate_hz', 50.0)
        # 제어 루프 주기 [Hz]

        # E-STOP 관련 파라미터 선언
        EStop.declare_parameters(self)

        # -------------------------
        # GAP FOLLOW
        # -------------------------
        self.declare_parameter('gf_speed', 2.0)
        # 기본 주행 속도 [m/s]

        self.declare_parameter('gf_max_steer', 0.4)
        # 최대 조향각 [rad]

        self.declare_parameter('gf_max_range', 10.0)
        # 라이다 최대 처리 거리 [m]

        self.declare_parameter('gf_front_angle_deg', 70.0)
        # gap-follow가 볼 전방 범위 [deg]

        self.declare_parameter('gf_min_gap_size', 8)
        # gap으로 인정할 최소 beam 개수

        self.declare_parameter('gf_min_valid_range', 0.05)
        # 이보다 작은 라이다 값은 무효 처리 [m]

        self.declare_parameter('gf_corner_speed_scale', 0.7)
        # 조향이 클수록 속도를 얼마나 줄일지 정하는 비율

        self.declare_parameter('gf_min_speed', 0.5)
        # 최소 속도 [m/s]

        self.declare_parameter('gf_close_distance', 2.0)
        # 장애물이 가까워졌다고 보는 거리 [m]

        self.declare_parameter('gf_very_close_distance', 1.0)
        # 매우 가까운 거리 [m]

        self.declare_parameter('gf_close_speed', 1.5)
        # 가까운 상황에서 허용할 속도 상한 [m/s]

        self.declare_parameter('gf_very_close_speed', 1.0)
        # 매우 가까운 상황에서 허용할 속도 상한 [m/s]

        # -------------------------
        # VEHICLE GEOMETRY
        # -------------------------
        self.declare_parameter('use_vehicle_geometry', True)
        # 차량 폭/여유를 실제 gap 계산에 반영할지 여부

        self.declare_parameter('vehicle_width', 0.30)
        # 차량 폭 [m]

        self.declare_parameter('vehicle_length', 0.50)
        # 차량 길이 [m]

        self.declare_parameter('vehicle_wheelbase', 0.33)
        # wheelbase [m]

        self.declare_parameter('vehicle_safety_margin', 0.05)
        # 추가 안전 여유 [m]

        # -------------------------
        # GAP / CORNER BIAS
        # -------------------------
        self.declare_parameter('gf_use_corner_bias', True)
        # 커브 방향 힌트 사용 여부

        self.declare_parameter('gf_center_sector_deg', 12.0)
        # 중앙 openness를 계산할 영역 각도 [deg]

        self.declare_parameter('gf_side_sector_start_deg', 20.0)
        # 좌/우 측 openness 계산 시작 각도 [deg]

        self.declare_parameter('gf_side_sector_end_deg', 70.0)
        # 좌/우 측 openness 계산 끝 각도 [deg]

        self.declare_parameter('gf_turn_open_ratio', 1.25)
        # 측면이 중앙보다 이 비율 이상 더 열려 있어야 turn hint로 인정

        self.declare_parameter('gf_turn_diff_min', 0.8)
        # 좌우 openness 차이가 이 값 이상이어야 방향 판정

        self.declare_parameter('gf_corner_bias_ratio', 0.78)
        # gap 내부에서 어느 쪽으로 bias할지 정하는 비율

        # -------------------------
        # ASYMMETRIC FOV
        # -------------------------
        self.declare_parameter('gf_use_asymmetric_fov', True)
        # 좌우 비대칭 FOV 사용 여부

        self.declare_parameter('gf_left_fov_deg_nominal', 70.0)
        # 평상시 왼쪽 FOV [deg]

        self.declare_parameter('gf_right_fov_deg_nominal', 70.0)
        # 평상시 오른쪽 FOV [deg]

        self.declare_parameter('gf_left_fov_deg_turn', 85.0)
        # turn hint가 왼쪽일 때 넓혀서 볼 각도 [deg]

        self.declare_parameter('gf_right_fov_deg_turn', 40.0)
        # turn hint가 왼쪽일 때 반대편 좁혀서 볼 각도 [deg]
        # 오른쪽 turn이면 반대로 적용됨

        # -------------------------
        # ONE-SIDE WALL FOLLOW PRIOR
        # -------------------------
        self.declare_parameter('race_follow_side', 'right')   # 'right' or 'left'
        # 어느 벽을 기준으로 따라갈지

        self.declare_parameter('wf_target_dist', 0.8)
        # 벽과 유지하고 싶은 목표 거리 [m]

        self.declare_parameter('wf_kp', 0.9)
        # wall-follow 비례 게인

        self.declare_parameter('wf_max_steer', 0.30)
        # wall-follow가 낼 수 있는 최대 조향 [rad]

        self.declare_parameter('wf_theta_deg', 50.0)
        # wall-follow 계산에 사용하는 두 beam 사이 각도 [deg]

        self.declare_parameter('wf_right_angle_b_deg', -90.0)
        # 오른쪽 벽 기준 beam 각도 [deg]

        self.declare_parameter('wf_left_angle_b_deg', 90.0)
        # 왼쪽 벽 기준 beam 각도 [deg]

        self.declare_parameter('wf_lookahead', 0.5)
        # wall-follow 예측 거리 [m]

        # wall-follow validity
        self.declare_parameter('wf_min_valid_dist', 0.15)
        # wall-follow에 사용할 최소 유효 거리 [m]

        self.declare_parameter('wf_max_valid_dist', 6.0)
        # wall-follow에 사용할 최대 유효 거리 [m]

        # -------------------------
        # HYBRID SWITCH CONDITIONS
        # -------------------------
        self.declare_parameter('hybrid_nearest_dist_threshold', 1.5)
        # 가장 가까운 장애물이 이보다 가까우면 gap 우선 [m]

        self.declare_parameter('hybrid_gap_width_threshold', 1.20)
        # gap 폭이 이보다 좁으면 gap 우선 [m]

        self.declare_parameter('hybrid_gap_override_steer_deg', 10.0)
        # gap 조향이 이보다 크면 gap 우선 [deg]

        self.declare_parameter('hybrid_disable_wall_on_turn_hint', True)
        # turn hint가 있으면 wall-follow 끌지 여부

        self.declare_parameter('hybrid_wall_weight', 1.0)
        # BLEND 모드에서 wall-follow 비중

        self.declare_parameter('hybrid_gap_weight', 1.0)
        # BLEND 모드에서 gap-follow 비중

        # -------------------------
        # STEERING SMOOTHING
        # -------------------------
        self.declare_parameter('gf_use_steer_rate_limit', True)
        # 조향 변화속도 제한 사용 여부

        self.declare_parameter('gf_max_steer_rate', 1.8)
        # 최대 조향 변화속도 [rad/s]

        # -------------------------
        # VISUALIZATION
        # -------------------------
        self.declare_parameter('viz_frame', 'base_link')
        # RViz 마커 기준 frame

        self.declare_parameter('viz_best_point_topic', '/gf/best_point_marker')
        # best point 마커 토픽

        self.declare_parameter('viz_nearest_point_topic', '/gf/nearest_point_marker')
        # nearest obstacle 마커 토픽

        self.declare_parameter('viz_bubble_topic', '/gf/bubble_marker')
        # bubble 마커 토픽

        self.declare_parameter('viz_gap_topic', '/gf/gap_marker')
        # gap 영역 마커 토픽

        self.declare_parameter('viz_footprint_topic', '/gf/footprint_marker')
        # 차량 footprint 마커 토픽

        self.declare_parameter('viz_fov_topic', '/gf/fov_marker')
        # 현재 탐색 FOV 마커 토픽

        # 파라미터를 짧게 읽기 위한 lambda
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
        # 가장 최근 라이다 메시지 저장

        self.odom = None
        # 가장 최근 오도메트리 저장

        self.prev_steer = 0.0
        # 이전 steering 값 (rate limit용)

        self.prev_time = None
        # 이전 loop 시각 (dt 계산용)

        self.estop = EStop(self)
        # 이 노드 안에서 사용할 EStop 객체 생성

        # =========================
        # ROS interfaces
        # =========================
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        # /scan 토픽 구독

        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        # /vesc/odom 토픽 구독

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            10
        )
        # 최종 조향/속도 명령 publish

        self.best_point_pub = self.create_publisher(Marker, self.viz_best_point_topic, 10)
        self.nearest_point_pub = self.create_publisher(Marker, self.viz_nearest_point_topic, 10)
        self.bubble_pub = self.create_publisher(Marker, self.viz_bubble_topic, 10)
        self.gap_pub = self.create_publisher(Marker, self.viz_gap_topic, 10)
        self.footprint_pub = self.create_publisher(Marker, self.viz_footprint_topic, 10)
        self.fov_pub = self.create_publisher(Marker, self.viz_fov_topic, 10)
        # RViz용 시각화 마커 publisher들

        self.create_timer(1.0 / self.control_rate_hz, self._loop)
        # 일정 주기로 _loop 실행

        self.get_logger().info('GapFollowNode ready (race-direction hybrid)')
        # 시작 로그

    def _scan_cb(self, msg):
        # scan 콜백
        # 최신 라이다 저장
        self.scan = msg

    def _odom_cb(self, msg):
        # odom 콜백
        # 최신 오도메트리 저장
        self.odom = msg

    def _loop(self):
        # 메인 제어 loop
        if self.scan is None or self.odom is None:
            # 아직 센서 데이터가 준비되지 않았으면 계산하지 않음
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        # 현재 시각 [s]

        if self.prev_time is None:
            dt = 1.0 / self.control_rate_hz
            # 첫 loop에서는 nominal dt 사용
        else:
            dt = now - self.prev_time
            # 실제 loop 간 시간 차 계산

            if dt <= 1e-4:
                dt = 1.0 / self.control_rate_hz
                # 너무 작은 dt는 비정상으로 보고 nominal 사용
            elif dt > 0.2:
                dt = 0.2
                # 너무 큰 dt는 상한 제한
        self.prev_time = now

        # 핵심 계산 수행
        result = self._compute(dt)

        # ROS 주행 명령 메시지 생성
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(result['steering'])
        msg.drive.speed = float(result['speed'])

        # 마지막에 EStop으로 안전 검사
        msg = self.estop.should_stop(self.scan, self.odom, msg)

        # 최종 명령 publish
        self.drive_pub.publish(msg)

        # RViz 시각화 publish
        self._publish_visualization(result)

    def _compute(self, dt):
        # gap following의 핵심 로직
        scan = self.scan
        ranges = np.array(scan.ranges, dtype=np.float32)

        # 라이다 데이터 정리
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        n = len(ranges)

        # 1) turn hint
        turn_hint = self._detect_turn_hint(ranges, angle_min, angle_inc, n)
        # 왼쪽이 더 열려 있는지, 오른쪽이 더 열려 있는지, 중앙인지 추정

        # 2) FOV selection
        left_fov_deg, right_fov_deg = self._get_fov_by_turn_hint(turn_hint)
        # turn hint에 따라 좌우 FOV를 다르게 설정

        start_idx = self._angle_to_index(-math.radians(right_fov_deg), angle_min, angle_inc, n)
        end_idx = self._angle_to_index(math.radians(left_fov_deg), angle_min, angle_inc, n)
        # 선택된 FOV를 라이다 index로 변환

        proc = ranges.copy()
        proc[:start_idx] = 0.0
        # FOV 바깥 왼쪽 영역 제거

        proc[end_idx + 1:] = 0.0
        # FOV 바깥 오른쪽 영역 제거

        proc[proc < self.min_valid_range] = 0.0
        # 너무 작은 값은 무효화

        # 3) nearest obstacle
        front_ranges = proc[start_idx:end_idx + 1]
        # 전방 FOV 내부 range만 추출

        valid = np.where(front_ranges > 0.0)[0]
        # 유효한 beam 인덱스 추출

        if len(valid) == 0:
            # 유효한 전방 데이터가 하나도 없으면 빈 결과 반환
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc)

        local_min = valid[np.argmin(front_ranges[valid])]
        # 전방에서 가장 가까운 장애물의 local index

        nearest_idx = start_idx + int(local_min)
        # 전체 scan 기준 index로 환산

        nearest_dist = float(proc[nearest_idx])
        # 가장 가까운 장애물 거리 [m]

        # 4) vehicle-width-based bubble
        effective_radius = self._get_effective_bubble_radius()
        # 차량 폭 + safety margin을 고려한 bubble 반지름 계산

        if nearest_dist > effective_radius:
            theta = math.atan2(effective_radius, nearest_dist)
            # 장애물 중심 기준 bubble이 차지하는 각도 계산
        else:
            theta = math.pi / 2.0
            # 너무 가까우면 bubble을 매우 넓게 처리

        bubble_size = int(theta / angle_inc)
        # bubble 각도를 beam 개수로 변환

        bubble_start = max(0, nearest_idx - bubble_size)
        bubble_end = min(n - 1, nearest_idx + bubble_size)
        # bubble 시작/끝 index 계산

        proc[bubble_start:bubble_end + 1] = 0.0
        # bubble 영역은 통과 불가로 보고 제거

        # 5) gap search
        free = np.where(proc > 0.0)[0]
        # bubble 제거 후 살아남은 free beam들

        if len(free) == 0:
            # free beam이 하나도 없으면 정지 결과 반환
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc, nearest_idx, nearest_dist, effective_radius)

        splits = np.split(free, np.where(np.diff(free) > 1)[0] + 1)
        # 연속된 beam들을 하나의 gap으로 분리

        gaps = [g for g in splits if len(g) >= self.min_gap_size]
        # 너무 작은 gap은 제거

        if len(gaps) == 0:
            # gap이 없으면 정지 결과 반환
            return self._empty_result(start_idx, end_idx, angle_min, angle_inc, nearest_idx, nearest_dist, effective_radius)

        best_gap = max(gaps, key=lambda g: len(g))
        # 가장 긴 gap 선택

        gap_start = int(best_gap[0])
        gap_end = int(best_gap[-1])

        # 6) gap-follow steer
        best_idx = self._select_best_point(gap_start, gap_end, turn_hint)
        # gap 내부에서 실제 목표 index 선택
        # 중앙 또는 코너 bias 적용 위치

        best_dist = float(proc[best_idx])
        # best point 거리

        best_angle = angle_min + best_idx * angle_inc
        # best point 각도 [rad]

        gap_steer = float(np.clip(best_angle, -self.max_steer, self.max_steer))
        # best_angle을 steering으로 사용하되 최대 조향각 제한

        gap_width_est = self._estimate_gap_width(gap_start, gap_end, max(best_dist, 0.8), angle_inc)
        # gap 폭을 대략적인 실거리로 추정

        # 7) one-side wall-follow prior
        wall_valid, wall_steer, wall_debug = self._compute_one_side_wall_follow(
            ranges, angle_min, angle_inc, n
        )
        # 한쪽 벽 기준 wall-follow steering 계산

        # 8) hybrid switching
        use_gap_override = False
        # 기본은 blend 가능 상태로 시작

        if nearest_dist < self.hybrid_nearest_dist_threshold:
            use_gap_override = True
            # 장애물이 너무 가까우면 gap-follow 우선

        if gap_width_est < self.hybrid_gap_width_threshold:
            use_gap_override = True
            # gap이 좁으면 gap-follow 우선

        if abs(math.degrees(gap_steer)) > self.hybrid_gap_override_steer_deg:
            use_gap_override = True
            # 큰 조향이 필요하면 gap-follow 우선

        if self.hybrid_disable_wall_on_turn_hint and turn_hint != 'center':
            use_gap_override = True
            # turn hint 있으면 wall-follow보다 gap-follow 우선

        if not wall_valid:
            use_gap_override = True
            # wall-follow 계산이 신뢰 불가면 gap-follow만 사용

        if use_gap_override:
            steering = gap_steer
            mode = 'GAP'
            # gap-follow만 사용
        else:
            steering = self.hybrid_wall_weight * wall_steer + self.hybrid_gap_weight * gap_steer
            mode = 'BLEND'
            # 둘을 가중합하여 사용

        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        # 최종 steering 제한

        if self.use_steer_rate_limit:
            steering = self._apply_steer_rate_limit(steering, dt)
            # 한 loop에서 조향이 너무 급하게 바뀌지 않게 제한

        # 9) speed
        steer_ratio = abs(steering) / self.max_steer if self.max_steer > 1e-6 else 0.0
        # 조향이 얼마나 큰지 정규화

        speed = self.gf_speed * (1.0 - self.corner_speed_scale * steer_ratio)
        # 조향이 클수록 속도를 줄임

        if best_dist < self.close_distance:
            speed = min(speed, self.close_speed)
            # 장애물이 가까우면 추가 감속

        if best_dist < self.very_close_distance:
            speed = min(speed, self.very_close_speed)
            # 매우 가까우면 더 강하게 감속

        speed = max(speed, self.min_speed)
        # 너무 느려지는 것 방지

        self.get_logger().info(
            f'[RACE] mode={mode} | steer={math.degrees(steering):.1f} deg | '
            f'gap_steer={math.degrees(gap_steer):.1f} deg | '
            f'wall_steer={math.degrees(wall_steer):.1f} deg | '
            f'nearest={nearest_dist:.2f} | gap_width={gap_width_est:.2f} | '
            f'turn_hint={turn_hint} | '
            f'wall_valid={wall_valid} | '
            f'wall_a={wall_debug["a"]:.2f} | wall_b={wall_debug["b"]:.2f}'
        )
        # 현재 의사결정 상태를 로그로 출력

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
        # 시각화 및 publish에 필요한 정보 함께 반환

    def _empty_result(self, start_idx, end_idx, angle_min, angle_inc,
                      nearest_idx=None, nearest_dist=None, effective_radius=None):
        # free gap이 없거나 데이터가 비정상일 때 반환하는 기본 결과
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
        # bubble 반지름 계산
        # 현재는 use_vehicle_geometry 여부와 관계없이
        # 차량폭/2 + safety_margin을 사용
        if self.use_vehicle_geometry:
            return self.vehicle_width / 2.0 + self.vehicle_safety_margin
        return self.vehicle_width / 2.0 + self.vehicle_safety_margin

    def _angle_to_index(self, angle_rad, angle_min, angle_inc, n):
        # 각도[rad]를 scan index로 변환
        idx = int((angle_rad - angle_min) / angle_inc)
        return max(0, min(n - 1, idx))
        # index 범위를 넘어가지 않게 clamp

    def _get_range_at(self, ranges, angle_min, angle_inc, angle_rad):
        # 특정 각도에서의 range를 보간(interpolation)하여 얻음
        f_idx = (angle_rad - angle_min) / angle_inc

        if f_idx < 0 or f_idx >= len(ranges) - 1:
            return 100.0
            # scan 범위 바깥이면 매우 큰 거리로 간주

        i0 = int(math.floor(f_idx))
        i1 = int(math.ceil(f_idx))

        r0 = ranges[i0]
        r1 = ranges[i1]

        if not np.isfinite(r0):
            r0 = 100.0
        if not np.isfinite(r1):
            r1 = 100.0
        # 비정상 값은 큰 거리로 치환

        w = f_idx - i0
        return float(r0 * (1.0 - w) + r1 * w)
        # 선형 보간

    def _sector_stat(self, ranges, angle_min, angle_inc, n, angle_start_deg, angle_end_deg):
        # 특정 각도 구간의 대표 openness 값을 계산
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
        # 중앙값을 사용해서 노이즈에 덜 민감하게 openness 측정

    def _detect_turn_hint(self, ranges, angle_min, angle_inc, n):
        # 좌/우/중앙 중 어느 쪽이 더 열려 있는지 판단
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
            # 왼쪽이 충분히 더 열려 있으면 left turn hint

        if right_score > center_score * self.turn_open_ratio and (right_score - left_score) > self.turn_diff_min:
            return 'right'
            # 오른쪽이 충분히 더 열려 있으면 right turn hint

        return 'center'
        # 특별히 한쪽이 더 열려 있지 않으면 center

    def _get_fov_by_turn_hint(self, turn_hint):
        # turn hint에 따라 좌우 FOV 결정
        if not self.use_asymmetric_fov:
            return self.left_fov_deg_nominal, self.right_fov_deg_nominal

        if turn_hint == 'left':
            return self.left_fov_deg_turn, self.right_fov_deg_turn
            # 왼쪽 turn이면 왼쪽 더 넓게, 오른쪽 더 좁게

        if turn_hint == 'right':
            return self.right_fov_deg_turn, self.left_fov_deg_turn
            # 오른쪽 turn이면 오른쪽 더 넓게, 왼쪽 더 좁게

        return self.left_fov_deg_nominal, self.right_fov_deg_nominal
        # 기본은 nominal FOV

    def _select_best_point(self, gap_start, gap_end, turn_hint):
        # gap 내부에서 실제 목표 beam 선택
        if gap_start is None or gap_end is None:
            return None

        gap_len = gap_end - gap_start

        if gap_len <= 0:
            return gap_start

        if turn_hint == 'center':
            return (gap_start + gap_end) // 2
            # 직진 상황이면 gap 중앙

        if turn_hint == 'left':
            idx = int(gap_start + self.corner_bias_ratio * gap_len)
            return max(gap_start, min(gap_end, idx))
            # 왼쪽 힌트면 gap 왼쪽 쪽으로 bias

        if turn_hint == 'right':
            idx = int(gap_start + (1.0 - self.corner_bias_ratio) * gap_len)
            return max(gap_start, min(gap_end, idx))
            # 오른쪽 힌트면 gap 오른쪽 쪽으로 bias

        return (gap_start + gap_end) // 2

    def _estimate_gap_width(self, gap_start, gap_end, rep_dist, angle_inc):
        # gap의 실제 폭을 대략 추정
        angular_width = max(0.0, (gap_end - gap_start) * angle_inc)
        return float(2.0 * rep_dist * math.sin(angular_width / 2.0))
        # 원호 기반 근사
        # rep_dist 위치에서 이 gap이 대략 몇 m 폭인지 추정

    def _compute_one_side_wall_follow(self, ranges, angle_min, angle_inc, n):
        # 한쪽 벽을 기준으로 한 wall-follow steering 계산
        theta = math.radians(self.wf_theta_deg)

        if self.race_follow_side == 'right':
            angle_b = math.radians(self.wf_right_angle_b_deg)
            # 오른쪽 벽 기준 beam
        else:
            angle_b = math.radians(self.wf_left_angle_b_deg)
            # 왼쪽 벽 기준 beam

        angle_a = angle_b + theta if self.race_follow_side == 'right' else angle_b - theta
        # 두 beam(a, b)를 사용해서 벽 방향 추정

        a = self._get_range_at(ranges, angle_min, angle_inc, angle_a)
        b = self._get_range_at(ranges, angle_min, angle_inc, angle_b)
        # a, b 거리 얻기

        valid = (
            self.wf_min_valid_dist < a < self.wf_max_valid_dist and
            self.wf_min_valid_dist < b < self.wf_max_valid_dist
        )
        # 두 값이 모두 유효 범위인지 검사

        if not valid:
            return False, 0.0, {'a': a, 'b': b}
            # 유효하지 않으면 wall-follow 무효

        denominator = a * math.sin(theta)

        if abs(denominator) < 1e-6:
            alpha = 0.0
            # 0으로 나누는 문제 방지
        else:
            numerator = a * math.cos(theta) - b
            alpha = math.atan2(numerator, denominator)
            # 벽의 기울기(alpha) 추정

        D_t = b * math.cos(alpha)
        # 현재 벽과의 수직 거리 추정

        D_t1 = D_t + self.wf_lookahead * math.sin(alpha)
        # lookahead 만큼 앞에서의 예상 거리

        error = self.wf_target_dist - D_t1
        # 목표 거리와의 오차

        # right wall 기준이면:
        # error > 0 -> 벽에서 멀다 -> 오른쪽으로 더 가야 함 -> 음의 steer
        if self.race_follow_side == 'right':
            raw_steer = -self.wf_kp * error
        else:
            raw_steer = self.wf_kp * error

        steer = float(np.clip(raw_steer, -self.wf_max_steer, self.wf_max_steer))
        # wall-follow steering 제한

        return True, steer, {'a': a, 'b': b}

    def _apply_steer_rate_limit(self, target_steer, dt):
        # 조향 변화속도 제한
        max_delta = self.max_steer_rate * dt
        # 한 loop에서 바뀔 수 있는 최대 steering 양

        delta = target_steer - self.prev_steer
        # 이번에 바꾸고 싶은 steering 변화량

        delta = max(-max_delta, min(max_delta, delta))
        # 변화량 제한

        limited = self.prev_steer + delta
        # 제한된 steering 계산

        self.prev_steer = limited
        # 다음 loop를 위해 저장

        return float(np.clip(limited, -self.max_steer, self.max_steer))

    def _polar_to_xy(self, idx, dist, angle_min, angle_inc):
        # scan index + 거리값을 x,y 좌표로 변환
        angle = angle_min + idx * angle_inc
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        return x, y

    def _make_marker_header(self, marker):
        # Marker 공통 header 설정
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = self.get_clock().now().to_msg()

    def _publish_visualization(self, result):
        # 모든 시각화 marker publish
        self._publish_vehicle_footprint()
        # 차량 footprint 표시

        self._publish_fov(result['start_idx'], result['end_idx'], result['angle_min'], result['angle_inc'])
        # 현재 탐색 FOV 표시

        if result['nearest_idx'] is not None and result['nearest_dist'] is not None:
            self._publish_nearest_point(result['nearest_idx'], result['nearest_dist'],
                                        result['angle_min'], result['angle_inc'])
            # 가장 가까운 장애물 표시

            self._publish_bubble(result['nearest_idx'], result['nearest_dist'],
                                 result['angle_min'], result['angle_inc'],
                                 result['effective_radius'])
            # bubble 표시

        if result['gap_start'] is not None and result['gap_end'] is not None:
            self._publish_gap(result['gap_start'], result['gap_end'],
                              result['angle_min'], result['angle_inc'])
            # 선택된 gap 영역 표시

        if result['best_idx'] is not None and result['best_dist'] is not None:
            self._publish_best_point(result['best_idx'], result['best_dist'],
                                     result['angle_min'], result['angle_inc'])
            # 최종 목표점 표시

    def _publish_best_point(self, idx, dist, angle_min, angle_inc):
        # best point를 초록 구로 표시
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
        # nearest obstacle를 빨간 구로 표시
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
        # bubble을 반투명 구로 표시
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
        # gap 전체를 노란 선으로 표시
        marker = Marker()
        self._make_marker_header(marker)

        marker.ns = 'gf_gap'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.04
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        rep_dist = min(3.0, self.max_range)
        # gap을 시각화할 대표 거리

        for idx in range(gap_start, gap_end + 1):
            x, y = self._polar_to_xy(idx, rep_dist, angle_min, angle_inc)

            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.02
            marker.points.append(p)

        self.gap_pub.publish(marker)

    def _publish_vehicle_footprint(self):
        # 차량 footprint를 자주색 선으로 표시
        half_w = self.vehicle_width / 2.0
        half_l = self.vehicle_length / 2.0

        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
            (half_l, half_w),
        ]
        # 사각형 꼭짓점

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
        # 현재 탐색하는 FOV 경계를 하늘색 선으로 표시
        marker = Marker()
        self._make_marker_header(marker)

        marker.ns = 'gf_fov'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.02
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)

        radius = min(3.0, self.max_range)
        # FOV 표시 길이

        origin = Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.01
        marker.points.append(origin)
        # 시작점을 원점으로 추가

        x1, y1 = self._polar_to_xy(start_idx, radius, angle_min, angle_inc)
        p1 = Point()
        p1.x = float(x1)
        p1.y = float(y1)
        p1.z = 0.01
        marker.points.append(p1)
        # 시작 경계선

        step = max(1, (end_idx - start_idx) // 30 + 1)
        # 점이 너무 많지 않도록 sampling 간격 설정

        for idx in range(start_idx, end_idx + 1, step):
            x, y = self._polar_to_xy(idx, radius, angle_min, angle_inc)
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.01
            marker.points.append(p)
        # FOV 곡선 부분

        x2, y2 = self._polar_to_xy(end_idx, radius, angle_min, angle_inc)
        p2 = Point()
        p2.x = float(x2)
        p2.y = float(y2)
        p2.z = 0.01
        marker.points.append(p2)
        # 끝 경계점

        marker.points.append(origin)
        # 다시 원점으로 닫아 부채꼴 형태로 보이게 함

        self.fov_pub.publish(marker)


def main(args=None):
    # ROS2 노드 실행 진입점
    rclpy.init(args=args)

    node = GapFollowNode()

    try:
        rclpy.spin(node)
        # 노드 실행
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 직전에 반드시 정지 명령을 한번 publish
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        node.drive_pub.publish(stop_msg)

        # 노드 종료
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()