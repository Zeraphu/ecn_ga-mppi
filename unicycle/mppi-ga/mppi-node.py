#!/usr/bin/env python3

"""
1. Converts transformed_global_path to numpy array for MPPI_GA
2. Performs MPPI_GA and recieves best_U and best_X
3. Converts best_U to Twist and publishes to cmd_vel_mppi
4. Converts best_X to PoseArray and publishes to /local_plan with /nav_msgs/msgs/Path message type
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import numpy as np
from ament_index_python.packages import get_package_share_directory

import MPPI_GA, motionModel, utils

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('mppi_node')

        self.horizon = 100
        self.num_paths = 100
        self.ref_path = np.empty((0, 3))
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.goal_reached = False
        self.goal_pose = np.array([np.inf, np.inf])

        # Initialize the turtle model with max velocity 0.2 and max angular velocity 0.4 with appropriate std deviations
        turtle = motionModel.turtleModel(
            dt=0.5,
            min_v=-0.1,
            max_v=0.26,
            max_w=0.2,
            v_std=0.013,
            w_std=0.091
        )
        
        self.mppiga_controller = MPPI_GA.MPPI_GA(self.horizon, self.num_paths, turtle)
        self.mppiga_controller.dynamics.generate_input(self.horizon)

        # Subscribers and Publishers
        qos_profile = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.transformed_global_path_subscriber = self.create_subscription(
            Path,
            '/transformed_global_plan',
            self.transformed_global_path_callback,
            10
        )

        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_sub,  # Reusing the same callback for simplicity
            10
        )
        
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel_mppi', qos_profile)
        self.local_plan_publisher = self.create_publisher(PoseArray, '/local_plan', qos_profile)

        self.timer = self.create_timer(0.1, self.mppi_callback)
        self.goal_checker = self.create_timer(0.5, self.check_goal_reached)
    
    def mppi_callback(self):
        if self.ref_path is None or len(self.ref_path) < 15:
            self.get_logger().info("Waiting for transformed global path...")
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist_msg)
            return

        elif len(self.ref_path) == 0:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist_msg)
            self.get_logger().info("Received empty reference path.")
            return
        elif self.current_state is None or len(self.current_state) == 0:
            self.get_logger().info("Current state is not set.")
            return
        
        elif self.ref_path is not None and len(self.ref_path) > 15 and not self.goal_reached:
            best_U, best_X, _, _ = self.mppiga_controller.run(self.current_state, self.ref_path)

            # Convert best_U to Twist message
            twist_msg = Twist()
            twist_msg.linear.x = best_U[0][0]
            twist_msg.angular.z = best_U[0][1]
            self.cmd_vel_publisher.publish(twist_msg)
            self.get_logger().info(f"Published Twist: linear.x={twist_msg.linear.x}, angular.z={twist_msg.angular.z}")

            # Convert best_X to PoseArray message
            pose_array_msg = PoseArray()
            pose_array_msg.header = Header()
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = 'map'
            pose_array_msg.poses = []
            for state in best_X:
                pose = Pose()
                pose.position.x = state[0]
                pose.position.y = state[1]
                pose.orientation.z = np.sin(state[2] / 2)
                pose.orientation.w = np.cos(state[2] / 2)
                pose_array_msg.poses.append(pose)
            self.local_plan_publisher.publish(pose_array_msg)

        elif self.goal_reached:
            self.get_logger().info("Goal has been reached, stopping the robot.")
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist_msg)

    def transformed_global_path_callback(self, msg):
        if msg is None or len(msg.poses) == 0:
            self.get_logger().info("Received empty or None transformed global path.")
            return
        
        self.ref_path = np.array([[pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.z] for pose in msg.poses])
        # self.get_logger().info(f"Received transformed global path with {len(self.ref_path)} points.")

        self.mppiga_controller.current_state = np.array([
            self.ref_path[0, 0],  # x
            self.ref_path[0, 1],  # y
            self.ref_path[0, 2]   # theta
        ])
        self.mppiga_controller.horizon = len(self.ref_path)-1
    
    def goal_pose_sub(self, msg):
        if msg is None:
            self.get_logger().info("Received None goal pose.")
            return
        
        self.goal_pose = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"\nReceived goal pose: x={self.goal_pose[0]}, y={self.goal_pose[1]}\n")
        self.check_goal_reached()

    def check_goal_reached(self,):
        distance_to_goal = np.sqrt((self.goal_pose[0] - self.mppiga_controller.current_state[0]) ** 2 \
                                    + (self.goal_pose[1] - self.mppiga_controller.current_state[1]) ** 2)
        
        self.get_logger().info(f"Goal checker: {self.goal_reached}")
        self.get_logger().info(f"Distance to goal: {distance_to_goal:.2f}")

        if distance_to_goal < 0.2:
            self.get_logger().info("Goal reached!")
            self.goal_reached = True
        else:
            self.goal_reached = False

def main(args=None):
    rclpy.init(args=args)
    node = MPPI_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()