#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import RobotState
from control_msgs.action import FollowJointTrajectory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class RebarSupervisor(Node):
    def __init__(self):
        super().__init__('rebar_supervisor')

        # --- Parameters ---
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('approach_distance', 0.15)
        self.declare_parameter('hover_time', 3.0)
        self.declare_parameter('ee_link', 'tool0')
        self.group         = self.get_parameter('planning_group').value
        self.approach_dist = self.get_parameter('approach_distance').value
        self.hover_time    = self.get_parameter('hover_time').value
        self.ee_link       = self.get_parameter('ee_link').value

        # --- State ---
        self.current_js = None
        self.is_busy    = False

        self.scan_joints = [0.7508, -1.7514, 1.8618, -0.1103, 0.8394, 0.0007]
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- I/O ---
        self.create_subscription(JointState, '/joint_states',        self.js_cb,        10)
        self.create_subscription(PoseArray,  '/rebar_intersections', self.intersect_cb, qos_reliable)

        self.active_index_pub   = self.create_publisher(Int32, '/active_rebar_index', 10)
        self.visit_num_pub      = self.create_publisher(Int32, '/rebar_visit_num',    10)
        self.detect_trigger_pub = self.create_publisher(Bool,  '/detect_intersections', 10)

        # --- Clients ---
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.exec_client = ActionClient(self, FollowJointTrajectory,
                                        '/arm_controller/follow_joint_trajectory')

        self.get_logger().info(f'[SUPERVISOR] Initializing... Planning Group: {self.group}')

        threading.Thread(target=self.auto_home, daemon=True).start()

    # ----------------- Callbacks -----------------
    def js_cb(self, msg):
        self.current_js = msg

    def intersect_cb(self, msg):
        if self.is_busy or not msg.poses:
            return
        self.get_logger().info(f'[SUPERVISOR] Received {len(msg.poses)} points. Starting sequence.')
        threading.Thread(target=self.execute_mission, args=(list(msg.poses),), daemon=True).start()

    # ----------------- Small publish helpers -----------------
    def _pub_index(self, idx):
        m = Int32(); m.data = int(idx); self.active_index_pub.publish(m)

    def _pub_visit(self, n):
        m = Int32(); m.data = int(n); self.visit_num_pub.publish(m)

    def _pub_trigger_detect(self):
        m = Bool(); m.data = True; self.detect_trigger_pub.publish(m)

    # ----------------- Mission flow -----------------
    def auto_home(self):
        self.exec_client.wait_for_server()
        while self.current_js is None:
            time.sleep(0.1)

        self.get_logger().info('[SUPERVISOR] Moving to Scan Pose...')
        self._pub_index(-1)
        self._pub_visit(0)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = self.scan_joints
        pt.time_from_start.sec = 5
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal(trajectory=traj)
        self.exec_client.send_goal_async(goal)
        time.sleep(6.0)

        self.get_logger().info('[SUPERVISOR] At Scan Pose. Triggering detection...')
        time.sleep(1.0)  # small settle so detector definitely has the latest frame
        self._pub_trigger_detect()
        self.get_logger().info('[SUPERVISOR] Detection trigger sent. Waiting for /rebar_intersections...')

    def execute_mission(self, poses):
        self.is_busy = True

        # Pair each pose with its ORIGINAL detector index so we can highlight
        # the right dot in the detector's storage while still visiting in snake order.
        indexed = list(enumerate(poses))
        ordered = self.order_snake_indexed(indexed)
        total   = len(ordered)

        for visit_num, (det_idx, target) in enumerate(ordered, start=1):
            self._pub_index(det_idx)        # which dot turns blue in the detector
            self._pub_visit(visit_num)      # what number shows in the counter

            self.get_logger().info(
                f'---> Visit {visit_num}/{total} (detector idx {det_idx}): Moving...'
            )

            approach = Pose()
            approach.position.x = target.position.x
            approach.position.y = target.position.y - self.approach_dist
            approach.position.z = target.position.z
            approach.orientation = self.ee_orientation_facing_plus_y()

            if self.cartesian_move(approach):
                self.get_logger().info(f'Reached visit {visit_num}. Hovering {self.hover_time}s.')
                time.sleep(self.hover_time)
            else:
                self.get_logger().error(f'Could not plan to visit {visit_num}. Skipping.')

        self.get_logger().info('=== MISSION COMPLETE ===')
        self._pub_index(-1)     # clear blue highlight
        self._pub_visit(-1)     # detector reads -1 as "MISSION COMPLETE"
        self.is_busy = False

    # ----------------- Motion -----------------
    def cartesian_move(self, target):
        if self.current_js is None:
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.header.stamp    = self.get_clock().now().to_msg()
        req.group_name      = self.group
        req.link_name       = self.ee_link
        req.max_step        = 0.01
        req.avoid_collisions = True
        req.start_state     = RobotState(joint_state=self.current_js)
        req.waypoints       = [target]

        future = self.cart_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        res = future.result()

        if not res or res.fraction < 0.9:  # tightened from 0.1 — partial paths were too lenient
            frac = res.fraction if res else 'None'
            self.get_logger().warn(f'Planning failed. Fraction: {frac}')
            return False

        goal = FollowJointTrajectory.Goal(trajectory=res.solution.joint_trajectory)
        send_goal_future = self.exec_client.send_goal_async(goal)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.01)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Action server rejected goal.')
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.01)
        return True

    # ----------------- Ordering & geometry -----------------
    def order_snake_indexed(self, indexed_poses):
        """Snake-order poses while preserving each one's original detector index.

        indexed_poses: list of (orig_idx, Pose)
        returns:       list of (orig_idx, Pose) in snake-visit order
        """
        sorted_by_z = sorted(indexed_poses, key=lambda ip: -ip[1].position.z)
        rows, z_tol, ordered = [], 0.08, []
        for ip in sorted_by_z:
            placed = False
            for r in rows:
                if abs(r[0][1].position.z - ip[1].position.z) < z_tol:
                    r.append(ip)
                    placed = True
                    break
            if not placed:
                rows.append([ip])
        for i, r in enumerate(rows):
            if i % 2 == 0:
                ordered.extend(sorted(r, key=lambda ip: -ip[1].position.x))
            else:
                ordered.extend(sorted(r, key=lambda ip:  ip[1].position.x))
        return ordered

    def ee_orientation_facing_plus_y(self):
        q = Quaternion()
        q.x, q.y, q.z, q.w = -0.7071, 0.0, 0.0, 0.7071
        return q


def main():
    rclpy.init()
    node = RebarSupervisor()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
