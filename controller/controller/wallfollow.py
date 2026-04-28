import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from controller.estop import EStop


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow')

        # ===== parameter declaration =====
        self.declare_parameter('control_rate_hz', 50.0)

        # E-STOP
        EStop.declare_parameters(self)

        # WALL FOLLOW
        self.declare_parameter('wf_target_dist', 0.8)
        self.declare_parameter('wf_kp', 1.0)
        self.declare_parameter('wf_kd', 0.1)
        self.declare_parameter('wf_ki', 0.01)
        self.declare_parameter('wf_speed', 1.5)
        self.declare_parameter('wf_max_steer', 0.4)
        self.declare_parameter('wf_lookahead', 0.5)
        self.declare_parameter('wf_min_speed', 0.55)

        # GEOMETRY
        self.declare_parameter('wf_theta_deg', 50.0)
        self.declare_parameter('wf_angle_b_deg', -90.0)
        self.declare_parameter('wf_front_angle_deg', 0.0)
        self.declare_parameter('wf_front_right_angle_deg', -20.0)

        # CORNER HANDLING
        self.declare_parameter('corner_a_threshold', 4.5)
        self.declare_parameter('corner_front_threshold', 1.5)
        self.declare_parameter('corner_b_threshold', 2.2)

        self.declare_parameter('corner_sharp_steer', -0.34)
        self.declare_parameter('corner_sharp_speed', 0.35)
        self.declare_parameter('corner_medium_steer', -0.28)
        self.declare_parameter('corner_medium_speed', 0.45)
        self.declare_parameter('corner_soft_steer', -0.18)
        self.declare_parameter('corner_soft_speed', 0.50)

        # CLAMP
        self.declare_parameter('integral_clip', 0.25)
        self.declare_parameter('derivative_clip', 5.0)

        p = lambda name: self.get_parameter(name).value

        # ===== load parameter values =====
        self.target_dist = float(p('wf_target_dist'))
        self.kp = float(p('wf_kp'))
        self.kd = float(p('wf_kd'))
        self.ki = float(p('wf_ki'))
        self.wf_speed = float(p('wf_speed'))
        self.lookahead = float(p('wf_lookahead'))
        self.max_steer = float(p('wf_max_steer'))
        self.wf_min_speed = float(p('wf_min_speed'))

        self.theta = math.radians(float(p('wf_theta_deg')))
        self.angle_b = math.radians(float(p('wf_angle_b_deg')))
        self.front_angle = math.radians(float(p('wf_front_angle_deg')))
        self.front_right_angle = math.radians(float(p('wf_front_right_angle_deg')))

        self.corner_a_threshold = float(p('corner_a_threshold'))
        self.corner_front_threshold = float(p('corner_front_threshold'))
        self.corner_b_threshold = float(p('corner_b_threshold'))

        self.corner_sharp_steer = float(p('corner_sharp_steer'))
        self.corner_sharp_speed = float(p('corner_sharp_speed'))
        self.corner_medium_steer = float(p('corner_medium_steer'))
        self.corner_medium_speed = float(p('corner_medium_speed'))
        self.corner_soft_steer = float(p('corner_soft_steer'))
        self.corner_soft_speed = float(p('corner_soft_speed'))

        self.integral_clip = float(p('integral_clip'))
        self.derivative_clip = float(p('derivative_clip'))

        self._prev_error = 0.0
        self._integral_error = 0.0
        self._prev_t = None
        self._debug_count = 0

        self.scan = None
        self.odom = None

        self.estop = EStop(self)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            10
        )

        self.create_timer(1.0 / float(p('control_rate_hz')), self._loop)

        self.get_logger().info('WallFollowNode ready')

    def _scan_cb(self, msg):
        self.scan = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (now - self._prev_t) if self._prev_t is not None else 0.02
        self._prev_t = now

        if dt <= 1e-4:
            dt = 0.02
        elif dt > 0.1:
            dt = 0.1

        steer, speed = self._compute(dt)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)

        msg = self.estop.should_stop(self.scan, self.odom, msg)
        self.drive_pub.publish(msg)

    def _get_range_at(self, angle_rad):
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        ranges = self.scan.ranges

        f_idx = (angle_rad - angle_min) / angle_inc

        if f_idx < 0 or f_idx >= len(ranges) - 1:
            return 100.0

        i0 = int(math.floor(f_idx))
        i1 = int(math.ceil(f_idx))

        if i0 < 0 or i1 >= len(ranges):
            return 100.0

        r0 = ranges[i0]
        r1 = ranges[i1]

        if not np.isfinite(r0):
            r0 = 100.0
        if not np.isfinite(r1):
            r1 = 100.0

        w = f_idx - i0
        r = r0 * (1.0 - w) + r1 * w
        return float(r)

    def _compute(self, dt):
        angle_a = self.angle_b + self.theta

        a = self._get_range_at(angle_a)
        b = self._get_range_at(self.angle_b)
        front = self._get_range_at(self.front_angle)
        front_right = self._get_range_at(self.front_right_angle)

        # ===== corner handling =====
        if a >= self.corner_a_threshold:
            self._integral_error = 0.0
            self._prev_error = 0.0

            if front < self.corner_front_threshold:
                steer = self.corner_sharp_steer
                speed = self.corner_sharp_speed
            elif b < self.corner_b_threshold:
                steer = self.corner_medium_steer
                speed = self.corner_medium_speed
            else:
                steer = self.corner_soft_steer
                speed = self.corner_soft_speed

            self._debug_count += 1
            if self._debug_count % 10 == 0:
                self.get_logger().info(
                    f"[CORNER] a={a:.2f}, b={b:.2f}, front={front:.2f}, "
                    f"front_right={front_right:.2f}, steer={steer:.2f}, speed={speed:.2f}"
                )
            return steer, speed

        # ===== normal wall follow =====
        denominator = a * math.sin(self.theta)

        if abs(denominator) < 1e-6:
            alpha = 0.0
        else:
            numerator = a * math.cos(self.theta) - b
            alpha = math.atan2(numerator, denominator)

        D_t = b * math.cos(alpha)
        D_t1 = D_t + self.lookahead * math.sin(alpha)

        error = self.target_dist - D_t1

        self._integral_error += error * dt
        self._integral_error = float(np.clip(
            self._integral_error, -self.integral_clip, self.integral_clip
        ))

        derivative = (error - self._prev_error) / dt if dt > 1e-6 else 0.0
        derivative = float(np.clip(
            derivative, -self.derivative_clip, self.derivative_clip
        ))

        raw_steer = (
            self.kp * error
            + self.ki * self._integral_error
            + self.kd * derivative
        )

        steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))

        steer_ratio = abs(steer) / self.max_steer if self.max_steer > 1e-6 else 0.0
        max_speed = self.wf_speed
        min_speed = self.wf_min_speed
        speed = max_speed - steer_ratio * (max_speed - min_speed)
        speed = float(np.clip(speed, min_speed, max_speed))

        self._prev_error = error

        self._debug_count += 1
        if self._debug_count % 10 == 0:
            self.get_logger().info(
                f"[FOLLOW] a={a:.2f}, b={b:.2f}, front={front:.2f}, front_right={front_right:.2f}, "
                f"alpha={alpha:.2f}, Dt={D_t:.2f}, Dt1={D_t1:.2f}, "
                f"error={error:.2f}, integ={self._integral_error:.2f}, deriv={derivative:.2f}, "
                f"steer={steer:.2f}, speed={speed:.2f}"
            )

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
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