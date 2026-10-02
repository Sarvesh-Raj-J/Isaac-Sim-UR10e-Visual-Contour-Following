#!/usr/bin/env python3
"""
seam_weld_supervisor.py
=======================
Seam welding supervisor for UR10e (eye-in-hand camera).

HOME_JOINTS = the wide-scan pose the user provided (what the robot sees in
the screenshot: box with 4 live edges detected).

Per-edge flow:
  1. FJT  → HOME_JOINTS            (reliable joint-space baseline)
  2. OMPL → above edge start       (position-only, no orientation constraint
                                    → IK always solvable, no sampling failures)
  3. FK   → read current EE quat   (use whatever orientation OMPL landed in)
  4. Cart → lower to seam start    (straight down)
  5. Cart → dense traverse to end  (line following, 4 mm steps)
  6. Cart → lift above seam end
  7. Repeat for next edge
  8. FJT  → HOME_JOINTS after all edges done
"""

import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, ColorRGBA, Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

# ── Config ─────────────────────────────────────────────────────────────────────
PLANNING_GROUP   = "arm"
EE_LINK          = "tool0"
REFERENCE_FRAME  = "base_link"

CART_MAX_STEP    = 0.004   # 4 mm — smooth seam line following
CART_JUMP_THRESH = 0.0
CART_MIN_FRAC    = 0.80
OMPL_APPROACH_FRAC = 0.70  # lenient for position-only approach (unused — OMPL not fraction-based)

TOOL_OFFSET   = 0.05   # metres above seam surface during traverse
APPROACH_LIFT = 0.12   # clearance above seam for approach / retreat

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# Wide-scan / home pose (from /joint_states echo — this IS the overview pose)
HOME_JOINTS = [-1.5007, -1.8652, -0.8619, 4.2768, 1.6200, -3.0813]


# ── Node ───────────────────────────────────────────────────────────────────────

class SeamWeldSupervisor(Node):

    def __init__(self):
        super().__init__("seam_weld_supervisor")

        self.declare_parameter("planning_group",    PLANNING_GROUP)
        self.declare_parameter("ee_link",           EE_LINK)
        self.declare_parameter("tool_offset",       TOOL_OFFSET)
        self.declare_parameter("approach_lift",     APPROACH_LIFT)
        self.declare_parameter("cart_min_fraction", CART_MIN_FRAC)

        self._group         = self.get_parameter("planning_group").value
        self._ee_link       = self.get_parameter("ee_link").value
        self._tool_offset   = self.get_parameter("tool_offset").value
        self._approach_lift = self.get_parameter("approach_lift").value
        self._cart_min_frac = self.get_parameter("cart_min_fraction").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._current_js: JointState | None = None
        self._is_busy = False
        self._js_lock = threading.Lock()
        self._home_quat: Quaternion | None = None  # FK result at HOME_JOINTS

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self._js_cb,    10)
        self.create_subscription(PoseArray,  "/seam_edges",   self._edges_cb, qos)

        # Publishers
        self._detect_pub = self.create_publisher(Bool,        "/detect_seams",      qos)
        self._active_pub = self.create_publisher(Int32,       "/active_seam_index",  10)
        self._phase_pub  = self.create_publisher(String,      "/scan_phase",          10)
        self._marker_pub = self.create_publisher(MarkerArray, "/seam_markers",        10)

        # Clients / action clients
        self._mg   = ActionClient(self, MoveGroup, "/move_action")
        self._cart = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._fk   = self.create_client(GetPositionFK,   "/compute_fk")
        self._fjt  = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        self.get_logger().info("[SUPERVISOR] Waiting for servers…")
        self._mg.wait_for_server(timeout_sec=30.0)
        self._fjt.wait_for_server(timeout_sec=30.0)
        self._cart.wait_for_service(timeout_sec=30.0)
        self._fk.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("[SUPERVISOR] All servers ready.")

        threading.Thread(target=self._pipeline, daemon=True).start()

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _js_cb(self, msg):
        with self._js_lock:
            self._current_js = msg

    def _edges_cb(self, msg: PoseArray):
        if self._is_busy or len(msg.poses) < 2:
            return
        edges = []
        for i in range(0, len(msg.poses) - 1, 2):
            edges.append((msg.poses[i], msg.poses[i + 1]))
        self.get_logger().info(f"[SUPERVISOR] {len(edges)} seam edges received.")
        self._publish_markers(edges)
        threading.Thread(target=self._weld_mission, args=(edges,), daemon=True).start()

    # ── Pipeline ───────────────────────────────────────────────────────────────
    def _pipeline(self):
        while self._current_js is None:
            time.sleep(0.1)

        # Arrive at home (wide-scan / overview pose)
        self.get_logger().info("[SUPERVISOR] Moving to home (wide-scan) pose…")
        self._pub_phase("OVERVIEW")
        self._pub_active(-1)
        self._joint_move(HOME_JOINTS, duration_sec=5)
        time.sleep(3.0)  # hold for overview display
        self.get_logger().info("[SUPERVISOR] Overview held. Capturing FK orientation…")

        # FK at HOME_JOINTS → reference orientation for any weld not reached by OMPL
        self._home_quat = self._get_ee_orientation()
        if self._home_quat:
            q = self._home_quat
            self.get_logger().info(
                f"[SUPERVISOR] Home EE quat: "
                f"x={q.x:.3f} y={q.y:.3f} z={q.z:.3f} w={q.w:.3f}"
            )
        else:
            self.get_logger().warn("[SUPERVISOR] FK failed — identity fallback.")
            self._home_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Trigger detection from the overview pose
        self._pub_phase("SCAN")
        time.sleep(0.5)
        self.get_logger().info("[SUPERVISOR] Triggering seam detection…")
        m = Bool(); m.data = True
        self._detect_pub.publish(m)
        self.get_logger().info("[SUPERVISOR] Waiting for /seam_edges…")

    # ── Welding mission ────────────────────────────────────────────────────────
    def _weld_mission(self, edges: list):
        self._is_busy = True
        total = len(edges)
        self.get_logger().info(f"[SUPERVISOR] ═══ Starting weld mission: {total} edges ═══")

        for idx, (start_pose, end_pose) in enumerate(edges):
            self._pub_active(idx)

            p0 = np.array([start_pose.position.x,
                           start_pose.position.y,
                           start_pose.position.z])
            p1 = np.array([end_pose.position.x,
                           end_pose.position.y,
                           end_pose.position.z])
            edge_len = float(np.linalg.norm(p1 - p0))

            self.get_logger().info(
                f"[SUPERVISOR] ══ Edge {idx+1}/{total}  len={edge_len:.3f}m\n"
                f"             p0={np.round(p0, 3)}  p1={np.round(p1, 3)}"
            )

            # 1. Return to HOME_JOINTS — always known state before each edge
            self.get_logger().info(f"  [1/3] Returning to HOME_JOINTS…")
            self._joint_move(HOME_JOINTS, duration_sec=4)
            time.sleep(0.3)

            # 2. OMPL position-only → above edge start
            above_start = p0 + np.array([0.0, 0.0, self._approach_lift])
            self.get_logger().info(
                f"  [2/3] OMPL approach → above_start={np.round(above_start, 3)}"
            )
            ok = self._ompl_position_move(above_start, f"edge {idx+1} approach")
            if not ok:
                self.get_logger().error(
                    f"  OMPL to {np.round(above_start,3)} failed — skipping edge {idx+1}."
                )
                continue

            # 3. Read actual EE orientation from FK after OMPL landed
            cur_q = self._get_ee_orientation() or self._home_quat
            self.get_logger().info(
                f"  [3/3] Cartesian weld: lower → traverse {edge_len*1000:.0f}mm → lift"
            )

            weld_start = self._pose(p0 + np.array([0.0, 0.0, self._tool_offset]), cur_q)
            weld_wps   = self._seam_waypoints(p0, p1, cur_q, self._tool_offset)
            above_end  = self._pose(p1 + np.array([0.0, 0.0, self._approach_lift]), cur_q)

            ok = self._cartesian_exec(
                [weld_start] + weld_wps + [above_end],
                f"edge {idx+1} weld ({len(weld_wps)+2} wps)",
                min_frac=self._cart_min_frac,
                max_step=CART_MAX_STEP,
            )
            if ok:
                self.get_logger().info(f"  ✓ Edge {idx+1} welded.")
            else:
                self.get_logger().warn(f"  ✗ Edge {idx+1} weld incomplete.")

            time.sleep(0.2)

        # Done — return home
        self._pub_active(-1)
        self._pub_phase("IDLE")
        self.get_logger().info("[SUPERVISOR] ═══ Welding mission complete ═══")
        self._joint_move(HOME_JOINTS, duration_sec=5)
        self._is_busy = False

    # ── OMPL: position-only (no orientation constraint) ────────────────────────
    def _ompl_position_move(self, xyz: np.ndarray, desc: str = "") -> bool:
        """
        Move to a 3-D position using OMPL with NO orientation constraint.
        This avoids 'Unable to sample valid states for goal tree' because
        the IK sampler can choose any valid wrist orientation.
        """
        target = Pose()
        target.position.x = float(xyz[0])
        target.position.y = float(xyz[1])
        target.position.z = float(xyz[2])
        target.orientation.w = 1.0   # dummy — not constrained

        pc = PositionConstraint()
        pc.header.frame_id = REFERENCE_FRAME
        pc.link_name       = self._ee_link
        pc.weight          = 1.0
        sp = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.015])
        bv = BoundingVolume()
        bv.primitives.append(sp)
        bv.primitive_poses.append(target)
        pc.constraint_region = bv

        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name                      = self._group
        r.pipeline_id                     = "ompl"
        r.planner_id                      = ""    # use default (RRTConnect)
        r.num_planning_attempts           = 15
        r.allowed_planning_time           = 10.0
        r.max_velocity_scaling_factor     = 0.3
        r.max_acceleration_scaling_factor = 0.2
        r.start_state.is_diff             = True
        r.goal_constraints                = [Constraints(position_constraints=[pc])]
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f"  OMPL position [{desc}] planning…")
        fut = self._mg.send_goal_async(goal)
        while rclpy.ok() and not fut.done():
            time.sleep(0.02)

        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error(f"  OMPL [{desc}]: goal rejected.")
            return False

        res = gh.get_result_async()
        while rclpy.ok() and not res.done():
            time.sleep(0.05)

        ec = res.result().result.error_code.val
        if ec != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"  OMPL [{desc}]: failed (code={ec}).")
            return False

        self.get_logger().info(f"  OMPL [{desc}]: reached position.")
        return True

    # ── Cartesian plan + execute ───────────────────────────────────────────────
    def _cartesian_exec(
        self,
        waypoints: list[Pose],
        desc: str = "",
        min_frac: float = CART_MIN_FRAC,
        max_step: float = CART_MAX_STEP,
    ) -> bool:
        with self._js_lock:
            js = self._current_js
        if js is None:
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id  = REFERENCE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = self._group
        req.link_name        = self._ee_link
        req.waypoints        = waypoints
        req.max_step         = max_step
        req.jump_threshold   = CART_JUMP_THRESH
        req.avoid_collisions = True
        req.start_state      = RobotState(joint_state=js)

        self.get_logger().info(
            f"  Cartesian [{desc}]: {len(waypoints)} wps, step={max_step*1000:.0f} mm…"
        )
        fut = self._cart.call_async(req)
        while rclpy.ok() and not fut.done():
            time.sleep(0.01)

        res = fut.result()
        if res is None or res.fraction < min_frac:
            frac = f"{res.fraction:.2f}" if res else "None"
            self.get_logger().warn(
                f"  Cartesian [{desc}]: fraction={frac} < {min_frac:.2f}."
            )
            return False

        self.get_logger().info(
            f"  Cartesian [{desc}]: fraction={res.fraction:.2f}. Executing…"
        )
        g = self._fjt.send_goal_async(
            FollowJointTrajectory.Goal(trajectory=res.solution.joint_trajectory)
        )
        while rclpy.ok() and not g.done():
            time.sleep(0.01)
        gh = g.result()
        if not gh or not gh.accepted:
            return False
        r = gh.get_result_async()
        while rclpy.ok() and not r.done():
            time.sleep(0.05)

        self.get_logger().info(f"  Cartesian [{desc}]: done.")
        return True

    # ── Joint-space move ───────────────────────────────────────────────────────
    def _joint_move(self, joints: list, duration_sec: int = 5):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(j) for j in joints]
        pt.time_from_start.sec = duration_sec
        traj.points.append(pt)

        fut = self._fjt.send_goal_async(
            FollowJointTrajectory.Goal(trajectory=traj)
        )
        while not fut.done():
            time.sleep(0.05)
        gh = fut.result()
        if gh and gh.accepted:
            res = gh.get_result_async()
            while not res.done():
                time.sleep(0.05)

    # ── FK: query current EE orientation ──────────────────────────────────────
    def _get_ee_orientation(self) -> Quaternion | None:
        with self._js_lock:
            js = self._current_js
        if js is None:
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = REFERENCE_FRAME
        req.fk_link_names   = [self._ee_link]
        req.robot_state     = RobotState(joint_state=js)

        fut = self._fk.call_async(req)
        while rclpy.ok() and not fut.done():
            time.sleep(0.01)

        res = fut.result()
        if (res is not None
                and hasattr(res, "error_code")
                and res.error_code.val == 1
                and res.pose_stamped):
            return res.pose_stamped[0].pose.orientation
        return None

    # ── Geometry helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _pose(xyz: np.ndarray, q: Quaternion) -> Pose:
        p = Pose()
        p.position.x  = float(xyz[0])
        p.position.y  = float(xyz[1])
        p.position.z  = float(xyz[2])
        p.orientation = q
        return p

    @staticmethod
    def _seam_waypoints(
        p0: np.ndarray, p1: np.ndarray, q: Quaternion, z_offset: float
    ) -> list[Pose]:
        """Dense intermediate poses along the seam, one every ~4 mm."""
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return []
        n    = max(2, int(L / 0.004))
        offs = np.array([0.0, 0.0, z_offset])
        wps  = []
        for i in range(1, n + 1):   # t=0 is already sent as weld_start
            pt = p0 + (i / n) * d + offs
            wps.append(SeamWeldSupervisor._pose(pt, q))
        return wps

    # ── Publisher helpers ──────────────────────────────────────────────────────
    def _pub_active(self, idx: int):
        m = Int32(); m.data = int(idx)
        self._active_pub.publish(m)

    def _pub_phase(self, phase: str):
        m = String(); m.data = phase
        self._phase_pub.publish(m)

    def _publish_markers(self, edges: list):
        ma = MarkerArray()
        for idx, (sp, ep) in enumerate(edges):
            m = Marker()
            m.header.frame_id = REFERENCE_FRAME
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "seam_edges"
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.007
            m.color           = ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0)
            m.lifetime.sec    = 0
            m.points = [
                Point(x=sp.position.x, y=sp.position.y,
                      z=sp.position.z + self._tool_offset),
                Point(x=ep.position.x, y=ep.position.y,
                      z=ep.position.z + self._tool_offset),
            ]
            ma.markers.append(m)
        self._marker_pub.publish(ma)


def main():
    rclpy.init()
    node = SeamWeldSupervisor()
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
