import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

PARAMS = {
    'control_rate_hz':         50.0,

    # ── Adaptive lookahead ──────────────────────────────────────────
    # lookahead = lerp(L_max → L_min)  as  kappa goes from 0 → kappa_corner_thresh
    'pp_lookahead_min':         0.8,   # [m]  lookahead at sharp corners
    'pp_lookahead_max':         3.0,   # [m]  lookahead on straights
    'pp_lookahead_preview':     1.5,   # [m]  distance ahead to scan for kappa

    # kappa >= corner_thresh → full corner mode (L_min, no boost)
    'pp_kappa_corner_thresh':   0.3,   # [rad/m]

    # ── Vehicle ─────────────────────────────────────────────────────
    'pp_wheelbase':             0.33,  # [m]
    'pp_max_steer':             0.4,   # [rad]

    # ── Speed ───────────────────────────────────────────────────────
    # speed_alpha: low-pass filter on commanded speed  (0=frozen, 1=instant)
    'pp_speed_alpha':           0.15,
    # straight_speed_boost: multiply waypoint vx on straights
    #   1.0 = use optimizer speed as-is;  1.2 = 20 % faster on straights
    'pp_straight_speed_boost':  1.0,
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.lookahead_min     = p('pp_lookahead_min')
        self.lookahead_max     = p('pp_lookahead_max')
        self.lookahead_preview = p('pp_lookahead_preview')
        self.kappa_corner_thresh = p('pp_kappa_corner_thresh')
        self.wheelbase         = p('pp_wheelbase')
        self.max_steer         = p('pp_max_steer')
        self.speed_alpha       = p('pp_speed_alpha')
        self.straight_boost    = p('pp_straight_speed_boost')

        self.odom       = None
        self.waypoints  = []
        self._cmd_speed = 0.0   # low-pass filtered commanded speed

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        self.drive_pub     = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('PPNode ready')

    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    # ------------------------------------------------------------------
    def _publish_lookahead(self, wp):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'pp_lookahead'; m.id = 0
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(wp.x_m)
        m.pose.position.y = float(wp.y_m)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        self.lookahead_pub.publish(m)

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return
        steer, speed = self._compute()
        msg = AckermannDriveStamped()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    # ------------------------------------------------------------------
    def _compute(self):
        """
        Curvature-adaptive Pure Pursuit.

        1. 차량 포즈 추출
        2. 전방 preview_dist 내 waypoint 중 최대 kappa 스캔
           → kappa_norm = clamp(max_kappa / kappa_corner_thresh, 0, 1)
        3. 적응형 lookahead = lerp(L_max, L_min, kappa_norm)
           코너일수록 짧게, 직선일수록 길게
        4. 목표 waypoint 탐색 (adaptive lookahead 거리에 가장 가까운 전방 포인트)
        5. 조향각 계산 (Pure Pursuit 기하학)
        6. 속도 = waypoint vx × boost(직선) → low-pass filter
        """

        # ── 1. 차량 포즈 ────────────────────────────────────────────
        pos   = self.odom.pose.pose.position
        px, py = pos.x, pos.y
        q     = self.odom.pose.pose.orientation
        yaw   = math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                           1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)

        N = len(self.waypoints)

        # ── 2. preview 구간에서 최대 kappa 스캔 ────────────────────
        # 가장 가까운 전방 waypoint를 찾고, 거기서부터 preview_dist만큼
        # 누적 거리로 전진하며 최대 |kappa|를 수집한다.
        closest_idx   = None
        closest_local = float('inf')
        for i, wp in enumerate(self.waypoints):
            dx = wp.x_m - px;  dy = wp.y_m - py
            lx = cos_y*dx - sin_y*dy
            if lx <= 0.0:
                continue
            d = math.sqrt(lx*lx + (sin_y*dx + cos_y*dy)**2)
            if d < closest_local:
                closest_local = d
                closest_idx   = i

        max_kappa_ahead = 0.0
        if closest_idx is not None:
            cum = 0.0
            idx = closest_idx
            for _ in range(min(80, N)):
                kappa = abs(self.waypoints[idx].kappa_radpm)
                if kappa > max_kappa_ahead:
                    max_kappa_ahead = kappa
                nxt  = (idx + 1) % N
                seg  = math.sqrt(
                    (self.waypoints[nxt].x_m - self.waypoints[idx].x_m)**2 +
                    (self.waypoints[nxt].y_m - self.waypoints[idx].y_m)**2
                )
                cum += seg
                if cum >= self.lookahead_preview:
                    break
                idx = nxt

        # ── 3. 적응형 lookahead 계산 ────────────────────────────────
        # kappa_norm: 0.0 = 완전 직선, 1.0 = 코너 임계값 이상
        kappa_norm = min(1.0, max_kappa_ahead / max(self.kappa_corner_thresh, 1e-6))
        lookahead  = self.lookahead_max + kappa_norm * (self.lookahead_min - self.lookahead_max)

        # ── 4. 목표 waypoint 탐색 ───────────────────────────────────
        best_idx  = None
        best_diff = float('inf')
        best_lx = best_ly = 0.0

        for i, wp in enumerate(self.waypoints):
            dx = wp.x_m - px;  dy = wp.y_m - py
            lx = cos_y*dx - sin_y*dy
            ly = sin_y*dx + cos_y*dy
            if lx <= 0.0:
                continue
            dist = math.sqrt(lx*lx + ly*ly)
            diff = abs(dist - lookahead)
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
                best_lx, best_ly = lx, ly

        if best_idx is None:
            return 0.0, 0.0

        self._publish_lookahead(self.waypoints[best_idx])

        # ── 5. Pure Pursuit 조향각 ──────────────────────────────────
        L2 = best_lx*best_lx + best_ly*best_ly
        if L2 < 1e-6:
            return 0.0, 0.0
        kappa_pp = 2.0 * best_ly / L2
        steer    = math.atan(self.wheelbase * kappa_pp)
        steer    = max(-self.max_steer, min(self.max_steer, steer))

        # ── 6. 속도 결정 ────────────────────────────────────────────
        # waypoint vx는 optimizer가 코너/직선 속도를 이미 계산한 값.
        # straight_boost > 1.0 이면 직선에서 optimizer 속도보다 빠르게.
        # kappa_norm=0 (직선) → boost 100% 적용
        # kappa_norm=1 (코너) → boost 0% 적용 (waypoint 속도 그대로)
        wp_speed     = float(self.waypoints[best_idx].vx_mps)
        if wp_speed <= 0.0:
            wp_speed = 1.5
        straight_ratio = 1.0 - kappa_norm
        target_speed   = wp_speed * (1.0 + (self.straight_boost - 1.0) * straight_ratio)

        # low-pass filter: 급격한 속도 명령 변화 완화
        self._cmd_speed += self.speed_alpha * (target_speed - self._cmd_speed)
        speed = self._cmd_speed

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = PPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
