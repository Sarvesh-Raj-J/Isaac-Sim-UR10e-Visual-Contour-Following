#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

class SimpleMove(Node):
    def __init__(self):
        super().__init__('simple_move_script')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('Waiting for move_action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Server available. Sending goal...')

    def send_goal(self):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "arm"
        goal_msg.request.max_velocity_scaling_factor = 0.1 # Slow for safety
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # Target Joint Values
        # Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        joint_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        
        # Values from user request
        joint_values = [
            4.71, 
            -1.4156, 
            1.4148, 
            -1.5269, 
            -1.6078, 
            -4.8149
        ]

        # Create Constraints
        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(constraints)

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        # self.get_logger().info(f'Received feedback: {feedback_msg.feedback.state}')
        pass

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        error_code = result.error_code.val
        if error_code == 1: # SUCCESS
            self.get_logger().info('Result: SUCCESS')
        else:
            self.get_logger().info(f'Result: FAILED with error code {error_code}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = SimpleMove()
    action_client.send_goal()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
