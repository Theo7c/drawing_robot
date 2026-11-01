#!/usr/bin/env python3
"""
arm_publisher.py
Streams a pre-calculated 4-joint trajectory at 50 Hz to /joint_states
and publishes a continuous marker trail of the drawing pen tip.
"""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point 

# Kinematic parameters matching your robot_arm.urdf (180 mm = 0.18 m)
L1 = 0.18
L2 = 0.18

class PlanarArmPublisher(Node):
    def __init__(self):
        super().__init__('planar_arm_publisher')

        # 1. Create Publishers
        # Queue size of 10 keeps latency low; old messages are dropped if unconsumed
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/pen_trail', 10)

        # 2. Load Trajectory
        # Expected shape: (N, 4) -> [theta1_R, theta2_R, theta1_L, theta2_L]
        try:
            self.trajectory = np.load('trajectory_angles.npy')
            self.num_points = len(self.trajectory)
            self.get_logger().info(f"Loaded trajectory with {self.num_points} waypoints.")
        except FileNotFoundError:
            self.get_logger().error("trajectory_angles.npy not found! Run ik_solver.py first.")
            sys.exit(1)

        # 3. Trajectory and Drawing State
        self.idx = 0
        self.trail_points = []

        # 4. Initialize 50 Hz Loop Timer
        # 1 / 50 Hz = 0.02 seconds = 20 milliseconds
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("50 Hz control loop running. Publishing to /joint_states...")

    def timer_callback(self):
        # Extract angles for the current time slice
        q = self.trajectory[self.idx]

        now = self.get_clock().now().to_msg()

        # ==========================================
        # 1. CONSTRUCT & PUBLISH JOINT STATE
        # ==========================================
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.header.frame_id = 'base_link'

        # Must match the joint names defined in robot_arm.urdf
        joint_msg.name = [
            'right_shoulder_joint',
            'right_elbow_joint',
            'left_shoulder_joint',
            'left_elbow_joint'
        ]

        # Angles must be standard Python floats (float64) in radians
        joint_msg.position = [
            float(q[0]),  # Right shoulder
            float(q[1]),  # Right elbow
            float(q[2]),  # Left shoulder
            float(q[3])   # Left elbow
        ]

        self.joint_pub.publish(joint_msg)

        # ==========================================
        # 2. CONSTRUCT & PUBLISH DRAWING TRAIL
        # ==========================================
        # Calculate pen tip (x, y) via Forward Kinematics on right arm chain
        theta_shoulder = q[0]
        theta_elbow = q[1]
        x_tip = L1 * np.cos(theta_shoulder) + L2 * np.cos(theta_shoulder + theta_elbow)
        y_tip = L1 * np.sin(theta_shoulder) + L2 * np.sin(theta_shoulder + theta_elbow)

        # Z is set just above the A4 paper surface (z = 0.003m)
        tip_pt = Point(x=float(x_tip), y=float(y_tip), z=0.003)
        self.trail_points.append(tip_pt)

        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = 'base_link'
        marker.ns = 'fourier_drawing'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.002  # 2 mm path stroke thickness

        # Dark grey/black ink color
        marker.color.r = 0.1
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 1.0  # Fully opaque

        marker.points = self.trail_points
        self.marker_pub.publish(marker)

        # ==========================================
        # 3. ADVANCE TRAJECTORY INDEX
        # ==========================================
        self.idx += 1
        if self.idx >= self.num_points:
            self.idx = 0
            self.trail_points.clear()  # Clear line trail when the loop resets

def main(args=None):
    rclpy.init(args=args)
    node = PlanarArmPublisher()

    try:
        # spin() blocks, servicing timer callbacks until Ctrl+C is pressed
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()