#!/usr/bin/env python3
"""
cube_edge_supervisor.py
=======================
Traverses all 12 edges of a detected cube using UR10e + MoveIt2.

Motion strategy — smooth, minimal HOME returns:
  1. FJT  → HOME_JOINTS                (once, at startup)
  2. OMPL → above first reachable edge  (once — gets robot into workspace)
  3. For each edge (chained, no HOME in between):
       ONE Cartesian call:
         current_pos (above_start)
           → at_start   (drop down, 4 mm steps)
           → traverse   (dense 4 mm steps along edge)
           → above_end  (lift up)
       Then for the NEXT edge, the Cartesian call prepends:
         via_clearance → above_next_start
       so robot glides straight to the next approach with no pause.
  4. Fallback: if Cartesian transit fails, OMPL re-plans to above that edge start.
  5. FJT  → HOME_JOINTS                (once, at end)

Unreachable edges (bottom face / occluded) are skipped silently.
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
    CollisionObject,
    Constraints,
    MoveItErrorCodes,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, ColorRGBA, Int32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

# ── Config ─────────────────────────────────────────────────────────────────────
PLANNING_GROUP  = "arm"
EE_LINK         = "tool0"
REFERENCE_FRAME = "base_link"

CART_MAX_STEP    = 0.004   # 4 mm — dense waypoints for smooth traversal
CART_JUMP_THRESH = 0.0
CART_MIN_FRAC    = 0.80

TOOL_OFFSET        = 0.02   # metres above edge surface during traverse
APPROACH_LIFT      = 0.10   # clearance above edge endpoints for approach / retreat
TRANSIT_EXTRA_LIFT = 0.05   # extra Z added to via-point between edges

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# Home (wide-scan) pose — matches seam_welding setup
HOME_JOINTS = [-1.5007, -1.8652, -0.8619, 4.2768, 1.6200, -3.0813]


class CubeEdgeSupervisor(Node):

    def __init__(self):
        super().__init__("cube_edge_supervisor")

        self.declare_parameter("planning_group",    PLANNING_GROUP)
        self.declare_parameter("ee_link",           EE_LINK)
        self.declare_parameter("tool_offset",       TOOL_OFFSET)
        self.declare_parameter("approach_lift",     APPROACH_LIFT)
        self.declare_parameter("cart_min_fraction", CART_MIN_FRAC)

        # Wall centre positions (base_link metres) — one (x,y,z) per wall face.
        # These match the X/Y/Z Position values shown in RViz → Scene Objects.
        self.declare_parameter("wall_left_x",    -0.81)
        self.declare_parameter("wall_left_y",     0.95)
        self.declare_parameter("wall_left_z",     0.725)
        self.declare_parameter("wall_right_x",    0.81)
        self.declare_parameter("wall_right_y",    0.95)
        self.declare_parameter("wall_right_z",    0.725)
        self.declare_parameter("wall_back_x",     0.00)
        self.declare_parameter("wall_back_y",     0.09)
        self.declare_parameter("wall_back_z",     0.725)
        self.declare_parameter("wall_front_x",    0.00)
        self.declare_parameter("wall_front_y",    1.81)
        self.declare_parameter("wall_front_z",    0.725)
        self.declare_parameter("wall_floor_x",    0.00)
        self.declare_parameter("wall_floor_y",    0.95)
        self.declare_parameter("wall_floor_z",   -0.06)
        self.declare_parameter("wall_ceiling_x",  0.00)
        self.declare_parameter("wall_ceiling_y",  0.95)
        self.declare_parameter("wall_ceiling_z",  1.51)

        self._group         = self.get_parameter("planning_group").value
        self._ee_link       = self.get_parameter("ee_link").value
        self._tool_offset   = self.get_parameter("tool_offset").value
        self._approach_lift = self.get_parameter("approach_lift").value
        self._cart_min_frac = self.get_parameter("cart_min_fraction").value
        self._walls = {
            name: (
                self.get_parameter(f"{name}_x").value,
                self.get_parameter(f"{name}_y").value,
                self.get_parameter(f"{name}_z").value,
            )
            for name in (
                "wall_left", "wall_right",
                "wall_back", "wall_front",
                "wall_floor", "wall_ceiling",
            )
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._current_js: JointState | None = None
        self._is_busy = False
        self._js_lock = threading.Lock()
        self._home_quat: Quaternion | None = None

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self._js_cb,    10)
        self.create_subscription(PoseArray,  "/cube_edges",   self._edges_cb, qos)

        # Publishers
        self._detect_pub = self.create_publisher(Bool,          "/detect_cube",      qos)
        self._active_pub = self.create_publisher(Int32,         "/active_cube_edge",  10)
        self._marker_pub = self.create_publisher(MarkerArray,   "/cube_edge_markers", 10)
        self._scene_pub  = self.create_publisher(PlanningScene, "/planning_scene",    10)

        # Clients
        self._mg   = ActionClient(self, MoveGroup, "/move_action")
        self._cart = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._fk   = self.create_client(GetPositionFK,   "/compute_fk")
        self._fjt  = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        self.get_logger().info("[CUBE_SUP] Waiting for action servers…")
        self._mg.wait_for_server(timeout_sec=30.0)
        self._fjt.wait_for_server(timeout_sec=30.0)
        self._cart.wait_for_service(timeout_sec=30.0)
        self._fk.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("[CUBE_SUP] All servers ready.")

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
        self.get_logger().info(f"[CUBE_SUP] {len(edges)} cube edges received.")
        self._publish_markers(edges)
        threading.Thread(target=self._traverse_mission, args=(edges,), daemon=True).start()

    # ── Startup pipeline ───────────────────────────────────────────────────────
    def _pipeline(self):
        while self._current_js is None:
            time.sleep(0.1)

        # Give the planning scene a moment to be ready, then add walls
        time.sleep(1.0)
        self._setup_workspace_walls()

        self.get_logger().info("[CUBE_SUP] Moving to HOME pose…")
        self._pub_active(-1)
        self._joint_move(HOME_JOINTS, duration_sec=5)
        time.sleep(3.0)

        # FK at HOME → fallback orientation
        self._home_quat = self._get_ee_orientation()
        if self._home_quat:
            q = self._home_quat
            self.get_logger().info(
                f"[CUBE_SUP] Home EE quat: "
                f"x={q.x:.3f} y={q.y:.3f} z={q.z:.3f} w={q.w:.3f}"
            )
        else:
            self.get_logger().warn("[CUBE_SUP] FK failed — using identity fallback.")
            self._home_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        time.sleep(0.5)
        self.get_logger().info("[CUBE_SUP] Triggering cube detection…")
        m = Bool(); m.data = True
        self._detect_pub.publish(m)
        self.get_logger().info("[CUBE_SUP] Waiting for /cube_edges…")

    # ── Traversal mission ──────────────────────────────────────────────────────
    def _traverse_mission(self, edges: list):
        self._is_busy = True
        total = len(edges)
        self.get_logger().info(
            f"[CUBE_SUP] ═══ Smooth traversal: {total} edges, single OMPL entry ═══"
        )

        # ── Phase 1: OMPL once to get into the workspace ──────────────────────
        # Try each edge in order; stop at the first one OMPL can reach.
        first_idx = None
        cur_q     = self._home_quat

        for candidate in range(total):
            p0c = np.array([edges[candidate][0].position.x,
                            edges[candidate][0].position.y,
                            edges[candidate][0].position.z])
            above_c = p0c + np.array([0.0, 0.0, self._approach_lift])
            self.get_logger().info(
                f"[CUBE_SUP] OMPL → above edge {candidate+1}…"
            )
            if self._ompl_position_move(above_c, f"entry edge {candidate+1}"):
                first_idx = candidate
                cur_q     = self._get_ee_orientation() or self._home_quat
                break

        if first_idx is None:
            self.get_logger().error("[CUBE_SUP] Cannot reach any edge. Aborting.")
            self._joint_move(HOME_JOINTS, duration_sec=5)
            self._is_busy = False
            return

        # ── Phase 2: Chain all edges with pure Cartesian ───────────────────────
        # prev_above_end tracks where the robot will be after finishing each edge
        # (= p1 + approach_lift of the last successfully started edge).
        prev_above_end: np.ndarray | None = None

        for idx in range(first_idx, total):
            self._pub_active(idx)
            sp, ep   = edges[idx]
            p0 = np.array([sp.position.x, sp.position.y, sp.position.z])
            p1 = np.array([ep.position.x, ep.position.y, ep.position.z])
            edge_len  = float(np.linalg.norm(p1 - p0))
            group     = "front" if idx < 4 else ("pillar" if idx < 8 else "back")
            above_start = p0 + np.array([0.0, 0.0, self._approach_lift])
            above_end   = p1 + np.array([0.0, 0.0, self._approach_lift])

            self.get_logger().info(
                f"[CUBE_SUP] ══ Edge {idx+1}/{total} [{group}]  {edge_len*1000:.0f} mm"
                f"\n             p0={np.round(p0,3)}  p1={np.round(p1,3)}"
            )

            edge_wps = self._edge_full_path(p0, p1, cur_q)

            if idx == first_idx:
                # Robot is already at above_start from the OMPL entry — just
                # send the edge path (approach-down → traverse → lift).
                ok = self._cartesian_exec(
                    edge_wps,
                    f"edge {idx+1} ({len(edge_wps)} wps)",
                    min_frac=self._cart_min_frac,
                    max_step=CART_MAX_STEP,
                )
            else:
                # Build a via-clearance point between prev_above_end and above_start
                # so the robot arcs smoothly overhead without clipping the cube.
                via      = (prev_above_end + above_start) / 2.0
                via[2]   = max(prev_above_end[2], above_start[2]) + TRANSIT_EXTRA_LIFT
                transit  = [self._pose(via,         cur_q),
                            self._pose(above_start, cur_q)]
                all_wps  = transit + edge_wps

                ok = self._cartesian_exec(
                    all_wps,
                    f"transit+edge {idx+1} ({len(all_wps)} wps)",
                    min_frac=0.65,       # generous — transit may shorten slightly
                    max_step=CART_MAX_STEP,
                )

                if not ok:
                    # Cartesian transit failed — fall back to OMPL for this edge only
                    self.get_logger().warn(
                        f"  Transit failed for edge {idx+1} — OMPL fallback"
                    )
                    if not self._ompl_position_move(above_start, f"edge {idx+1} fallback"):
                        self.get_logger().error(f"  Cannot reach edge {idx+1} — skipping.")
                        prev_above_end = above_end.copy()
                        continue
                    cur_q = self._get_ee_orientation() or cur_q
                    ok = self._cartesian_exec(
                        self._edge_full_path(p0, p1, cur_q),
                        f"edge {idx+1} post-OMPL ({len(edge_wps)} wps)",
                        min_frac=self._cart_min_frac,
                        max_step=CART_MAX_STEP,
                    )

            if ok:
                self.get_logger().info(f"  ✓ Edge {idx+1} done.")
            else:
                self.get_logger().warn(f"  ✗ Edge {idx+1} incomplete.")

            # Update state for next iteration regardless of success
            cur_q          = self._get_ee_orientation() or cur_q
            prev_above_end = above_end.copy()

        # ── Phase 3: Return HOME ───────────────────────────────────────────────
        self._pub_active(-1)
        self.get_logger().info("[CUBE_SUP] ═══ Traversal complete — returning HOME ═══")
        self._joint_move(HOME_JOINTS, duration_sec=5)
        self._is_busy = False

    # ── OMPL: position-only approach ───────────────────────────────────────────
    def _ompl_position_move(self, xyz: np.ndarray, desc: str = "") -> bool:
        target = Pose()
        target.position.x = float(xyz[0])
        target.position.y = float(xyz[1])
        target.position.z = float(xyz[2])
        target.orientation.w = 1.0

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
        r.planner_id                      = ""
        r.num_planning_attempts           = 15
        r.allowed_planning_time           = 10.0
        r.max_velocity_scaling_factor     = 0.3
        r.max_acceleration_scaling_factor = 0.2
        r.start_state.is_diff             = True
        r.goal_constraints                = [Constraints(position_constraints=[pc])]
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f"  OMPL [{desc}] planning…")
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

    def _edge_full_path(
        self, p0: np.ndarray, p1: np.ndarray, q: Quaternion
    ) -> list[Pose]:
        """
        Dense waypoints for one complete edge motion starting from above_start
        (robot is already there):

          at_start     p0 + [0,0,tool_offset]          ← approach-down
          wp_1 … wp_N  interpolated in 4 mm steps       ← traverse
          above_end    p1 + [0,0,approach_lift]          ← lift-off

        All waypoints share the same orientation q so the wrist stays fixed
        throughout the whole approach-traverse-retreat sequence.
        """
        offs = np.array([0.0, 0.0, self._tool_offset])
        lift = np.array([0.0, 0.0, self._approach_lift])
        d    = p1 - p0
        L    = float(np.linalg.norm(d))

        if L < 1e-4:
            # Degenerate edge — just hover at it
            return [self._pose(p0 + lift, q)]

        n   = max(2, int(L / CART_MAX_STEP))   # one dot every 4 mm
        wps = []

        # 1. Drop to edge surface at start
        wps.append(self._pose(p0 + offs, q))

        # 2. Traverse: n dots along the edge, one every 4 mm
        for i in range(1, n + 1):
            wps.append(self._pose(p0 + (i / n) * d + offs, q))

        # 3. Lift above edge end
        wps.append(self._pose(p1 + lift, q))

        return wps

    # ── Workspace collision walls ──────────────────────────────────────────────
    def _setup_workspace_walls(self, wall_thickness: float = 0.02):
        """
        Add 6 thin box collision objects to the planning scene.
        Centre positions come directly from the wall_*_x/y/z parameters
        (matching the X/Y/Z values visible in RViz → Scene Objects).
        Box dimensions are derived from the distance between opposing walls.
        """
        w = self._walls
        lx, rx = w["wall_left"][0],    w["wall_right"][0]
        by, fy = w["wall_back"][1],    w["wall_front"][1]
        fz, cz = w["wall_floor"][2],   w["wall_ceiling"][2]
        dx = rx - lx
        dy = fy - by
        dz = cz - fz
        t  = wall_thickness

        # (name, centre_xyz, box_size_xyz)
        walls = [
            ("wall_left",    w["wall_left"],    (t,  dy, dz)),
            ("wall_right",   w["wall_right"],   (t,  dy, dz)),
            ("wall_back",    w["wall_back"],    (dx, t,  dz)),
            ("wall_front",   w["wall_front"],   (dx, t,  dz)),
            ("wall_floor",   w["wall_floor"],   (dx, dy, t )),
            ("wall_ceiling", w["wall_ceiling"], (dx, dy, t )),
        ]

        ps = PlanningScene()
        ps.is_diff = True

        for name, centre, size in walls:
            co = CollisionObject()
            co.header.frame_id = REFERENCE_FRAME
            co.header.stamp    = self.get_clock().now().to_msg()
            co.id              = name
            co.operation       = CollisionObject.ADD

            box = SolidPrimitive(type=SolidPrimitive.BOX,
                                 dimensions=list(size))
            p = Pose()
            p.position.x, p.position.y, p.position.z = centre
            p.orientation.w = 1.0

            co.primitives.append(box)
            co.primitive_poses.append(p)
            ps.world.collision_objects.append(co)

        self._scene_pub.publish(ps)
        self.get_logger().info(
            f"[CUBE_SUP] Workspace walls set: "
            f"left_x={lx} right_x={rx} back_y={by} front_y={fy} "
            f"floor_z={fz} ceiling_z={cz}"
        )

    # ── Publisher helpers ──────────────────────────────────────────────────────
    def _pub_active(self, idx: int):
        m = Int32(); m.data = int(idx)
        self._active_pub.publish(m)

    def _publish_markers(self, edges: list):
        ma = MarkerArray()
        colors = {
            "front":  ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0),   # cyan
            "pillar": ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0),   # orange
            "back":   ColorRGBA(r=0.6, g=0.0, b=1.0, a=1.0),   # purple
        }
        for idx, (sp, ep) in enumerate(edges):
            group = "front" if idx < 4 else ("pillar" if idx < 8 else "back")
            m = Marker()
            m.header.frame_id = REFERENCE_FRAME
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = "cube_edges"
            m.id              = idx
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.006
            m.color           = colors[group]
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
    node = CubeEdgeSupervisor()
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
