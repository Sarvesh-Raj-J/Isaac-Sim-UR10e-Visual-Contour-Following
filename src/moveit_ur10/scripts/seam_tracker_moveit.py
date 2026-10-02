#!/usr/bin/env python3
"""
seam_tracker_moveit.py
======================
Receives 3D seam edges from seam_detector via /seam_edges and traverses them.

Mirrors the rebar_supervisor pipeline exactly:
  1. Move to scan pose (FollowJointTrajectory, fast)
  2. Trigger /detect_seams → detector publishes /seam_edges
  3. For each detected edge (OMPL approach → Cartesian traverse)
  4. Back to scan pose

Motion:
  - MoveGroup action  (pipeline_id="ompl")  → approach moves between edges
  - GetCartesianPath  → linear tracking along each seam edge
  - FollowJointTrajectory → execution of both

Launch via:
  ros2 launch moveit_ur10 seam_tracker.launch.py
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseArray

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLANNING_GROUP   = "arm"
EE_LINK          = "tool0"
REFERENCE_FRAME  = "base_link"
PLANNER_ID       = "RRTConnectkConfigDefault"
PLANNING_TIME    = 5.0
PLAN_ATTEMPTS    = 5
VEL_SCALE        = 0.25
ACC_SCALE        = 0.15
CART_MAX_STEP    = 0.005
CART_JUMP_THRESH = 0.0
CART_MIN_FRAC    = 0.80
TOOL_OFFSET      = 0.05    # metres above seam
APPROACH_LIFT    = 0.12    # extra Z lift for OMPL approach

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# Scan / home joint configuration  (matches initial_positions.yaml)
SCAN_JOINTS = [-1.4583, -1.9741, -1.0906, -1.5558, 1.6094, -3.11]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tool_down_quat() -> Quaternion:
    """180° around X  →  tool0 z-axis points down into the workpiece."""
    return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


def make_joint_constraints(joint_values: list) -> Constraints:
    c = Constraints()
    for name, val in zip(JOINT_NAMES, joint_values):
        jc = JointConstraint()
        jc.joint_name     = name
        jc.position       = float(val)
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight         = 1.0
        c.joint_constraints.append(jc)
    return c


def make_pose_constraints(pose: Pose) -> Constraints:
    pc = PositionConstraint()
    pc.header.frame_id = REFERENCE_FRAME
    pc.link_name       = EE_LINK
    sp     = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
    region = BoundingVolume()
    region.primitives.append(sp)
    region.primitive_poses.append(pose)
    pc.constraint_region = region
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = REFERENCE_FRAME
    oc.link_name       = EE_LINK
    oc.orientation     = pose.orientation
    oc.absolute_x_axis_tolerance = 0.5
    oc.absolute_y_axis_tolerance = 0.5
    oc.absolute_z_axis_tolerance = 0.5
    oc.weight = 0.5

    return Constraints(position_constraints=[pc], orientation_constraints=[oc])


def edge_waypoints(p0: np.ndarray, p1: np.ndarray) -> list:
    """Dense Cartesian waypoints along the edge, lifted by TOOL_OFFSET."""
    d  = p1 - p0
    L  = np.linalg.norm(d)
    if L < 1e-6:
        return []
    n  = max(6, int(L / CART_MAX_STEP))
    q  = _tool_down_quat()
    lift = np.array([0.0, 0.0, TOOL_OFFSET])
    wps = []
    for t in np.linspace(0.0, 1.0, n):
        pt = p0 + t * d + lift
        wps.append(Pose(
            position=Point(x=pt[0], y=pt[1], z=pt[2]),
            orientation=q,
        ))
    return wps


# ─────────────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────────────

class SeamTrackerMoveIt(Node):

    def __init__(self):
        super().__init__("seam_tracker")

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.current_js: JointState | None = None
        self.is_busy                        = False
        self._js_lock                       = threading.Lock()

        # Subscriptions
        self.create_subscription(JointState, "/joint_states",  self._js_cb,    10)
        self.create_subscription(PoseArray,  "/seam_edges",    self._edges_cb, qos)

        # Publishers
        self.detect_pub       = self.create_publisher(Bool,  "/detect_seams",     qos)
        self.active_seam_pub  = self.create_publisher(Int32, "/active_seam_index", 10)
        self.marker_pub       = self.create_publisher(MarkerArray, "/seam_markers", 10)

        # MoveGroup action → OMPL plan+execute
        self._mg  = ActionClient(self, MoveGroup,            "/move_action")
        # GetCartesianPath → seam edge interpolation
        self._cart = self.create_client(GetCartesianPath,    "/compute_cartesian_path")
        # FollowJointTrajectory → execute cartesian result + scan-pose move
        self._fjt  = ActionClient(self, FollowJointTrajectory,
                                  "/arm_controller/follow_joint_trajectory")

        self.get_logger().info("[TRACKER] Waiting for servers…")
        self._mg.wait_for_server(timeout_sec=30.0)
        self._cart.wait_for_service(timeout_sec=30.0)
        self._fjt.wait_for_server(timeout_sec=30.0)
        self.get_logger().info("[TRACKER] All servers ready.")

        # Start the scan→detect flow in a background thread
        threading.Thread(target=self._auto_scan, daemon=True).start()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _js_cb(self, msg):
        with self._js_lock:
            self.current_js = msg

    def _edges_cb(self, msg: PoseArray):
        if self.is_busy or len(msg.poses) < 2:
            return
        # Pair up consecutive poses into edge start/end
        edges = []
        for i in range(0, len(msg.poses) - 1, 2):
            edges.append((msg.poses[i], msg.poses[i + 1]))
        self.get_logger().info(f"[TRACKER] Received {len(edges)} seam edges.")
        self._publish_markers(edges)
        threading.Thread(target=self._execute_mission,
                         args=(edges,), daemon=True).start()

    # ── Auto scan: go to scan pose then trigger detector ───────────────────────
    def _auto_scan(self):
        self._fjt.wait_for_server()
        # Wait for first joint state
        while self.current_js is None:
            time.sleep(0.1)

        self.get_logger().info("[TRACKER] Moving to scan pose…")
        self._pub_active(-1)

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = SCAN_JOINTS
        pt.time_from_start.sec = 5
        traj.points.append(pt)

        fut = self._fjt.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj))
        while not fut.done():
            time.sleep(0.05)
        gh = fut.result()
        if gh.accepted:
            res = gh.get_result_async()
            while not res.done():
                time.sleep(0.05)

        self.get_logger().info("[TRACKER] At scan pose. Triggering detector…")
        time.sleep(1.0)   # settle so camera gets a clean frame
        m = Bool(); m.data = True
        self.detect_pub.publish(m)
        self.get_logger().info("[TRACKER] Detection triggered — waiting for /seam_edges…")

    # ── Mission: traverse every detected edge ─────────────────────────────────
    def _execute_mission(self, edges: list):
        self.is_busy = True
        total = len(edges)

        for idx, (start_pose, end_pose) in enumerate(edges):
            self._pub_active(idx)
            p0 = np.array([start_pose.position.x,
                           start_pose.position.y,
                           start_pose.position.z])
            p1 = np.array([end_pose.position.x,
                           end_pose.position.y,
                           end_pose.position.z])
            self.get_logger().info(
                f"[TRACKER] ─── Edge {idx+1}/{total}  "
                f"{np.round(p0,3)} → {np.round(p1,3)}"
            )

            # Approach pose: above edge start, tool pointing down
            ap = Pose()
            ap.position.x   = p0[0]
            ap.position.y   = p0[1]
            ap.position.z   = p0[2] + TOOL_OFFSET + APPROACH_LIFT
            ap.orientation  = _tool_down_quat()

            if not self._ompl_move(make_pose_constraints(ap), f"approach {idx+1}"):
                self.get_logger().warn(f"  OMPL approach failed — skipping edge {idx+1}.")
                continue

            time.sleep(0.2)

            wps = edge_waypoints(p0, p1)
            if not wps:
                continue

            self._cartesian_move(wps, f"edge {idx+1}")
            time.sleep(0.3)

        self._pub_active(-1)
        self.get_logger().info("[TRACKER] === Mission complete ===")
        self.is_busy = False

    # ── OMPL (MoveGroup action, plan+execute) ─────────────────────────────────
    def _ompl_move(self, constraints: Constraints, desc: str = "") -> bool:
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name                      = PLANNING_GROUP
        r.pipeline_id                     = "ompl"
        r.planner_id                      = PLANNER_ID
        r.num_planning_attempts           = PLAN_ATTEMPTS
        r.allowed_planning_time           = PLANNING_TIME
        r.max_velocity_scaling_factor     = VEL_SCALE
        r.max_acceleration_scaling_factor = ACC_SCALE
        r.start_state.is_diff             = True
        r.goal_constraints                = [constraints]
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f"  OMPL [{desc}] planning…")
        fut = self._mg.send_goal_async(goal)
        while rclpy.ok() and not fut.done():
            time.sleep(0.02)

        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error(f"  OMPL [{desc}]: goal rejected.")
            return False

        res = gh.get_result_async()
        while rclpy.ok() and not res.done():
            time.sleep(0.05)

        ec = res.result().result.error_code.val
        if ec != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"  OMPL [{desc}]: failed (code={ec}).")
            return False

        self.get_logger().info(f"  OMPL [{desc}]: done.")
        return True

    # ── Cartesian path + execution ────────────────────────────────────────────
    def _cartesian_move(self, waypoints: list, desc: str = "") -> bool:
        with self._js_lock:
            js = self.current_js
        if js is None:
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id  = REFERENCE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = PLANNING_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = waypoints
        req.max_step         = CART_MAX_STEP
        req.jump_threshold   = CART_JUMP_THRESH
        req.avoid_collisions = True
        req.start_state      = RobotState(joint_state=js)

        self.get_logger().info(
            f"  Cartesian [{desc}] planning {len(waypoints)} waypoints…"
        )
        fut = self._cart.call_async(req)
        while rclpy.ok() and not fut.done():
            time.sleep(0.01)

        res = fut.result()
        if res is None or res.fraction < CART_MIN_FRAC:
            frac = res.fraction if res else "None"
            self.get_logger().warn(
                f"  Cartesian [{desc}]: incomplete (fraction={frac:.2f}). Skipping."
            )
            return False

        self.get_logger().info(
            f"  Cartesian [{desc}]: fraction={res.fraction:.2f}. Executing…"
        )
        goal = FollowJointTrajectory.Goal(
            trajectory=res.solution.joint_trajectory
        )
        g = self._fjt.send_goal_async(goal)
        while rclpy.ok() and not g.done():
            time.sleep(0.01)
        gh = g.result()
        if not gh.accepted:
            return False
        r = gh.get_result_async()
        while rclpy.ok() and not r.done():
            time.sleep(0.05)

        self.get_logger().info(f"  Cartesian [{desc}]: done.")
        return True

    # ── Publisher helpers ─────────────────────────────────────────────────────
    def _pub_active(self, idx: int):
        m = Int32(); m.data = int(idx)
        self.active_seam_pub.publish(m)

    def _publish_markers(self, edges: list):
        """Orange LINE_STRIP markers in RViz for each detected seam edge."""
        ma = MarkerArray()
        for idx, (sp, ep) in enumerate(edges):
            m = Marker()
            m.header.frame_id = REFERENCE_FRAME
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns    = "seam_edges"
            m.id    = idx
            m.type  = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.007
            m.color   = ColorRGBA(r=1.0, g=0.35, b=0.0, a=1.0)
            m.lifetime.sec = 0
            m.points = [
                Point(x=sp.position.x, y=sp.position.y,
                      z=sp.position.z + TOOL_OFFSET),
                Point(x=ep.position.x, y=ep.position.y,
                      z=ep.position.z + TOOL_OFFSET),
            ]
            ma.markers.append(m)
        self.marker_pub.publish(ma)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = SeamTrackerMoveIt()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
