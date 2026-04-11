import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from controller.estop import EStop

PARAMS = {
    'control_rate_hz': 50.0,
    'pp_lookahead':     1.0,
    'pp_wheelbase':    0.33,
    'pp_max_steer':     0.4,
    'pp_heading_threshold_deg': 45.0,
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop     = EStop(self)
        self.lookahead = p('pp_lookahead')
        self.wheelbase = p('pp_wheelbase')
        self.max_steer = p('pp_max_steer')
        self.heading_threshold = math.radians(p('pp_heading_threshold_deg'))

        self.scan      = None
        self.odom      = None
        self.waypoints = []
        self.wp_x      = np.empty(0, dtype=np.float64)
        self.wp_y      = np.empty(0, dtype=np.float64)
        self.wp_v      = np.empty(0, dtype=np.float64)
        self.target_x  = 0.0
        self.target_y  = 0.0

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        from sensor_msgs.msg import LaserScan
        self.create_subscription(LaserScan,  '/scan',             self._scan_cb, 10)
        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.waypoints_marker_pub = self.create_publisher(Marker, '/pp/waypoints_marker', 10)
        self.target_marker_pub = self.create_publisher(Marker, '/pp/target_marker', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('PPNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        self.wp_x = np.fromiter((wp.x_m for wp in self.waypoints), dtype=np.float64, count=len(self.waypoints))
        self.wp_y = np.fromiter((wp.y_m for wp in self.waypoints), dtype=np.float64, count=len(self.waypoints))
        self.wp_v = np.fromiter((wp.vx_mps for wp in self.waypoints), dtype=np.float64, count=len(self.waypoints))

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

        if self.scan is not None and self.estop.is_stop_required(self.scan, self.odom):
            steer, speed = 0.0, 0.0
        else:
            steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)
        self._publish_markers()

    def _compute(self):
        pose = self.odom.pose.pose
        pos = pose.position
        ori = pose.orientation

        yaw = math.atan2(
            2.0 * (ori.w * ori.z + ori.x * ori.y),
            1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z),
        )

        dx = self.wp_x - pos.x
        dy = self.wp_y - pos.y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy
        dist2 = dx * dx + dy * dy
        lookahead2 = self.lookahead * self.lookahead

        heading_error = np.abs(np.arctan2(y_local, x_local))
        candidates = (x_local > 0.0) & (heading_error <= self.heading_threshold)
        if np.any(candidates):
            idx = int(np.argmin(np.where(candidates, np.abs(dist2 - lookahead2), np.inf)))
        else:
            idx = int(np.argmin(dist2))

        self.target_x = float(self.wp_x[idx])
        self.target_y = float(self.wp_y[idx])
        target_y = y_local[idx]
        target_dist2 = max(dist2[idx], 1e-6)
        steer = math.atan2(2.0 * self.wheelbase * target_y, target_dist2)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        speed = max(0.0, float(self.wp_v[idx]))
        return steer, speed

    def _publish_markers(self):
        if self.wp_x.size == 0:
            return

        stamp = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'pp_waypoints'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.points = [
            Point(x=float(x), y=float(y), z=0.0)
            for x, y in zip(self.wp_x, self.wp_y)
        ]
        self.waypoints_marker_pub.publish(marker)

        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.header.stamp = stamp
        target_marker.ns = 'pp_target'
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = self.target_x
        target_marker.pose.position.y = self.target_y
        target_marker.pose.position.z = 0.0
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.scale.z = 0.2
        target_marker.color.a = 1.0
        target_marker.color.r = 1.0
        self.target_marker_pub.publish(target_marker)


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
