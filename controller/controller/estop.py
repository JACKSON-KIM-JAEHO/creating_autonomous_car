import math                      # 각도 변환, cos 계산 등에 사용
import numpy as np              # 라이다 배열을 빠르게 처리하기 위해 사용


class EStop:
    # E-Stop 기능을 담당하는 클래스
    # 역할:
    # 1) 너무 가까운 장애물이 있으면 즉시 정지
    # 2) TTC(Time To Collision)가 위험하면 정지
    # 3) 필요하면 조향각 방향으로 검사 중심을 이동하여 커브에서 오탐 감소

    def __init__(self, node):
        # node: 이 EStop을 사용하는 ROS2 노드 객체
        # 여기서는 gap_follow 노드가 전달됨

        self._logger = node.get_logger()   # 경고/정보 로그를 출력하기 위한 logger 저장

        # 파라미터를 짧게 읽기 위한 함수
        # 예: p('estop_stop_distance') -> node.get_parameter('estop_stop_distance').value
        p = lambda name: node.get_parameter(name).value

        # -------------------------
        # E-Stop 기본 파라미터 읽기
        # -------------------------
        self.stop_distance = float(p('estop_stop_distance'))
        # 이 거리[m]보다 가까운 장애물이 있으면 즉시 정지

        self.estop_fov_deg = float(p('estop_fov_deg'))
        # 전방 감시 시야각[deg]
        # 예: 30도면 중심 기준 좌우 15도 범위를 검사

        self.ttc_threshold = float(p('estop_ttc_threshold'))
        # TTC 임계값[s]
        # 최소 TTC가 이 값보다 작으면 정지

        self.max_range = float(p('estop_max_range'))
        # 라이다 최대 처리 거리[m]
        # inf 값은 이 값으로 바꿔서 처리

        self.min_speed = float(p('estop_min_speed'))
        # 현재 속도가 이 값[m/s]보다 작으면 TTC 검사 의미가 적으므로 그냥 통과

        # -------------------------
        # 동적 중심 이동 파라미터
        # -------------------------
        self.use_dynamic_center = bool(p('estop_use_dynamic_center'))
        # True면 현재 조향각 방향으로 검사 중심을 이동

        self.center_gain = float(p('estop_center_gain'))
        # 조향각에 몇 배를 곱해 검사 중심을 이동할지 결정

        self.max_center_shift_deg = float(p('estop_max_center_shift_deg'))
        # 검사 중심 이동 최대 한계[deg]

        # 직전 loop에서 EStop이 걸렸는지 기록
        self._was_triggered = False

    @staticmethod
    def declare_parameters(node):
        # 이 함수는 EStop 객체를 생성하기 전에
        # 필요한 파라미터들을 ROS2 parameter로 선언하기 위한 함수

        node.declare_parameter('estop_stop_distance', 0.5)
        # 기본 즉시정지 거리[m]

        node.declare_parameter('estop_fov_deg', 30.0)
        # 기본 전방 시야각[deg]

        node.declare_parameter('estop_ttc_threshold', 0.35)
        # 기본 TTC threshold[s]

        node.declare_parameter('estop_max_range', 10.0)
        # 라이다 최대 처리 거리[m]

        node.declare_parameter('estop_min_speed', 0.1)
        # 최소 속도[m/s]

        # dynamic center 관련 기본값
        node.declare_parameter('estop_use_dynamic_center', True)
        # 동적 중심 사용 여부

        node.declare_parameter('estop_center_gain', 0.65)
        # 조향각 반영 gain

        node.declare_parameter('estop_max_center_shift_deg', 35.0)
        # 최대 중심 이동각[deg]

    def should_stop(self, scan, odom, cmd):
        """
        scan : LaserScan
        odom : Odometry
        cmd  : AckermannDriveStamped

        역할:
        - 현재 주행 명령(cmd)을 입력받아
        - 위험하면 speed를 0.0으로 바꾼 뒤 반환
        - 안전하면 그대로 반환
        """

        # scan / odom / cmd 중 하나라도 아직 안 들어왔으면 판단 불가
        # 이 경우 현재 명령을 그대로 반환
        if scan is None or odom is None or cmd is None:
            return cmd

        # 현재 명령 속도 읽기
        speed = float(cmd.drive.speed)

        # 너무 저속이면 TTC 검사 의미가 적고 노이즈에 민감해질 수 있음
        # 그래서 이 경우는 그대로 통과
        if abs(speed) < self.min_speed:
            self._was_triggered = False
            return cmd

        # -------------------------
        # 라이다 데이터 배열로 변환
        # -------------------------
        ranges = np.array(scan.ranges, dtype=np.float32)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        # NaN은 0으로 처리
        ranges[np.isnan(ranges)] = 0.0

        # 무한대(inf)는 max_range로 바꿔 처리
        ranges[np.isinf(ranges)] = self.max_range

        # 값 범위를 [0, max_range]로 제한
        ranges = np.clip(ranges, 0.0, self.max_range)

        # =========================
        # Dynamic FOV center
        # =========================
        center_angle = 0.0
        # 기본 검사 중심은 정면(0 rad)

        if self.use_dynamic_center:
            # 현재 조향각 읽기
            steer = float(cmd.drive.steering_angle)

            # 조향각에 비례해서 검사 중심 이동
            # 예: 오른쪽 조향이면 오른쪽으로 중심 이동
            center_angle = self.center_gain * steer

            # 너무 많이 이동하지 않도록 최대 한계 적용
            max_shift = math.radians(self.max_center_shift_deg)
            center_angle = float(np.clip(center_angle, -max_shift, max_shift))

        # 전체 FOV의 반 각도
        half_fov = math.radians(self.estop_fov_deg) / 2.0

        # 각 라이다 beam의 절대 각도 계산
        beam_angles = angle_min + np.arange(len(ranges)) * angle_inc

        # 검사 중심 기준 상대 각도 계산
        rel_angles = beam_angles - center_angle

        # 각도를 -pi ~ pi 범위로 정규화
        # 이유: 각도 wrap-around 문제 방지
        rel_angles = np.arctan2(np.sin(rel_angles), np.cos(rel_angles))

        # 검사 중심 기준 좌우 half_fov 안에 있는 beam만 선택
        mask = np.abs(rel_angles) <= half_fov

        masked_ranges = ranges[mask]
        masked_rel_angles = rel_angles[mask]

        # 검사할 beam이 하나도 없으면 정지 판단하지 않고 통과
        if len(masked_ranges) == 0:
            self._was_triggered = False
            return cmd

        # =========================
        # 1) immediate stop distance
        # =========================
        # 유효한 거리값(0보다 큰 값)만 추출
        positive = masked_ranges[masked_ranges > 0.0]

        if len(positive) > 0:
            # 가장 가까운 거리
            min_dist = float(np.min(positive))

            # 즉시정지 거리보다 가까우면 바로 정지
            if min_dist < self.stop_distance:
                stop_cmd = cmd
                stop_cmd.drive.speed = 0.0
                # speed만 0으로 바꾸고 steering은 그대로 둠

                self._was_triggered = True

                self._logger.warn(
                    f'EStop triggered (distance) | speed={speed:.2f}, '
                    f'min_dist={min_dist:.2f}, center={math.degrees(center_angle):.1f} deg'
                )
                # 어떤 이유로 멈췄는지 로그 출력

                return stop_cmd

        # =========================
        # 2) TTC check
        # =========================
        forward_speed = abs(speed)
        # 현재는 전진/후진 부호를 구분하지 않고 절댓값 사용

        valid = (masked_ranges > 0.0)
        # 유효한 거리만 다시 선택

        if np.any(valid):
            rr = masked_ranges[valid]
            # 유효한 거리 배열

            aa = masked_rel_angles[valid]
            # 각 거리값에 대응하는 상대 각도 배열

            # 각 beam 방향으로의 접근 속도
            # 정면이면 cos(0)=1 -> 접근속도 큼
            # 옆 방향이면 cos 값이 작아짐
            closing_speed = forward_speed * np.cos(aa)

            # 실제로 접근하고 있는 경우만 사용
            positive_closing = closing_speed > 1e-3

            if np.any(positive_closing):
                # TTC = 거리 / 접근속도
                ttc = rr[positive_closing] / closing_speed[positive_closing]

                # 가장 위험한(가장 작은) TTC
                min_ttc = float(np.min(ttc))

                # TTC가 threshold보다 작으면 정지
                if min_ttc < self.ttc_threshold:
                    stop_cmd = cmd
                    stop_cmd.drive.speed = 0.0

                    self._was_triggered = True

                    self._logger.warn(
                        f'EStop triggered (ttc) | speed={speed:.2f}, '
                        f'min_ttc={min_ttc:.2f}, center={math.degrees(center_angle):.1f} deg'
                    )

                    return stop_cmd

        # 거리 기반, TTC 기반 모두 걸리지 않으면 통과
        self._was_triggered = False
        return cmd