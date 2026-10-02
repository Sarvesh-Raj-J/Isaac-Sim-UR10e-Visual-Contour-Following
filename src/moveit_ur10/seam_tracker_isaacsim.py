"""
seam_tracker_isaacsim.py  —  Isaac Sim standalone
==================================================
UR10e end-effector tracks the seam edges of a geometric shape.
No welding physics — only the path-tracking motion is simulated.

Supported shapes:  cube | box | l_shape | triangle_prism
You can add any shape by defining vertices + edge pairs in SHAPES below.

Run with Isaac Sim's python:
    /home/robotics/isaac-sim-5.0/python.sh seam_tracker_isaacsim.py
    /home/robotics/isaac-sim-5.0/python.sh seam_tracker_isaacsim.py --shape box
    /home/robotics/isaac-sim-5.0/python.sh seam_tracker_isaacsim.py --shape l_shape --headless
"""

import argparse
import os
import sys
import tempfile

# ── Parse args before SimulationApp so --headless can be passed in ────────────
parser = argparse.ArgumentParser(description="UR10e seam tracker in Isaac Sim")
parser.add_argument("--shape",    default="cube",
                    choices=["cube", "box", "l_shape", "triangle_prism"],
                    help="Workpiece shape whose edges to track")
parser.add_argument("--size",     type=float, default=0.30,
                    help="Characteristic size of the workpiece (metres)")
parser.add_argument("--height",   type=float, default=0.05,
                    help="Tool offset above the seam line (metres)")
parser.add_argument("--density",  type=int,   default=12,
                    help="Waypoints sampled per edge")
parser.add_argument("--speed",    type=int,   default=4,
                    help="Simulation frames held per waypoint (lower = faster)")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# ── Bootstrap Isaac Sim ───────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})

import numpy as np
from scipy.spatial.transform import Rotation

import carb
import omni.kit.commands
import omni.usd
from pxr import UsdLux, Sdf, Gf

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils import viewports
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
)

# ── File paths ────────────────────────────────────────────────────────────────
_WS        = "/home/robotics/seam_ws/src"
_URDF_SRC  = f"{_WS}/ur10e_description/urdf/ur10e.urdf"
_MESH_ROOT = f"{_WS}/ur10e_description"
_ROBOT_DESC = (
    "/home/robotics/isaac-sim-5.0/exts/"
    "isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/universal_robots/ur10e/rmpflow/"
    "ur10e_robot_description.yaml"
)
_ROBOT_PRIM = "/World/ur10e"
_EE_FRAME   = "tool0"            # end-effector link in the URDF

# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE DEFINITIONS
#  Each shape returns a list of (v0, v1) seam edges in world-space coordinates.
#  Add new shapes here — the rest of the script is shape-agnostic.
# ─────────────────────────────────────────────────────────────────────────────

# Workpiece is placed 60 cm in front of the robot base, centred left-right,
# with its bottom face sitting on the table surface.
_TABLE_Z  = 0.10   # table surface height (metres above world origin)
_REACH_X  = 0.60   # distance in front of the robot base


def _offset(verts: np.ndarray, size: float) -> np.ndarray:
    """Translate shape so its bottom is on the table and centred at _REACH_X."""
    z_min = verts[:, 2].min()
    cx    = verts[:, 0].mean()
    cy    = verts[:, 1].mean()
    return verts + np.array([_REACH_X - cx, -cy, _TABLE_Z - z_min])


def cube_seams(size: float):
    h = size / 2
    v = np.array([
        [-h, -h, 0],  # 0 bottom-back-left
        [ h, -h, 0],  # 1 bottom-back-right
        [ h,  h, 0],  # 2 bottom-front-right
        [-h,  h, 0],  # 3 bottom-front-left
        [-h, -h, size],  # 4 top-back-left
        [ h, -h, size],  # 5 top-back-right
        [ h,  h, size],  # 6 top-front-right
        [-h,  h, size],  # 7 top-front-left
    ])
    v = _offset(v, size)
    edges = [
        (0,1),(1,2),(2,3),(3,0),       # bottom face
        (4,5),(5,6),(6,7),(7,4),       # top face
        (0,4),(1,5),(2,6),(3,7),       # vertical pillars
    ]
    return [(v[i], v[j]) for i, j in edges]


def box_seams(size: float):
    lx, ly, lz = size, size * 0.6, size * 0.8
    v = np.array([
        [0, 0, 0],   [lx, 0, 0],   [lx, ly, 0],  [0, ly, 0],
        [0, 0, lz],  [lx, 0, lz],  [lx, ly, lz], [0, ly, lz],
    ])
    v = _offset(v, size)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(v[i], v[j]) for i, j in edges]


def l_shape_seams(size: float):
    s = size
    # L cross-section: two rectangles joined at a corner
    v2d = np.array([
        [0,   0  ],
        [s,   0  ],
        [s,   s*0.4],
        [s*0.4, s*0.4],
        [s*0.4, s  ],
        [0,   s  ],
    ])
    depth = s * 0.5
    verts = []
    for z in [0, depth]:
        for (x, y) in v2d:
            verts.append([x, y, z])
    v = np.array(verts)
    v = _offset(v, size)
    n = len(v2d)
    edges  = [(i, (i+1) % n) for i in range(n)]             # bottom face
    edges += [(i+n, (i+1) % n + n) for i in range(n)]       # top face
    edges += [(i, i+n) for i in range(n)]                    # vertical edges
    return [(v[i], v[j]) for i, j in edges]


def triangle_prism_seams(size: float):
    r = size / 2
    h = size * 0.8
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    v2d = [(r * np.cos(a), r * np.sin(a)) for a in angles]
    v = np.array([[x, y, z] for z in [0, h] for (x, y) in v2d])
    v = _offset(v, size)
    n = 3
    edges  = [(i, (i+1) % n) for i in range(n)]
    edges += [(i+n, (i+1) % n + n) for i in range(n)]
    edges += [(i, i+n) for i in range(n)]
    return [(v[i], v[j]) for i, j in edges]


SHAPES = {
    "cube":           cube_seams,
    "box":            box_seams,
    "l_shape":        l_shape_seams,
    "triangle_prism": triangle_prism_seams,
}


# ─────────────────────────────────────────────────────────────────────────────
#  SEAM → WAYPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def seam_to_waypoints(seam_edges, points_per_edge: int, tool_offset: float):
    """
    Convert edge list to (position, orientation) waypoints for the TCP.

    Orientation convention:
      - Tool z-axis points straight down (-world Z) so the torch faces into
        the workpiece for all horizontal and near-horizontal seams.
      - For near-vertical edges the tool tilts slightly toward the seam.

    Returns list of (pos [3], quat_wxyz [4]).
    """
    # "Torch pointing down" quaternion in wxyz:
    # 180° around world-X maps z=[0,0,1] → z=[0,0,-1]
    _DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # w,x,y,z

    waypoints = []
    for (p0, p1) in seam_edges:
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            continue
        edge_dir = edge_vec / edge_len

        # Decide tool orientation
        z_world = np.array([0.0, 0.0, 1.0])
        vertical_component = abs(np.dot(edge_dir, z_world))

        if vertical_component > 0.85:
            # Mostly vertical edge — tilt torch 45° toward edge
            side = np.array([1.0, 0.0, 0.0])
            tilt_axis = np.cross(z_world, side)
            tilt_axis /= np.linalg.norm(tilt_axis)
            R_tilt = Rotation.from_rotvec(np.pi / 4 * tilt_axis)
            R_down = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])   # xyzw: 180° around x
            R_full = R_tilt * R_down
            q_xyzw = R_full.as_quat()
            quat   = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        else:
            quat = _DOWN_QUAT

        # Sample points evenly along edge; lift by tool_offset in world Z
        for t in np.linspace(0.0, 1.0, points_per_edge):
            pos = p0 + t * edge_vec + np.array([0.0, 0.0, tool_offset])
            waypoints.append((pos.copy(), quat.copy()))

    return waypoints


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: patch package:// URIs so Isaac Sim can find meshes
# ─────────────────────────────────────────────────────────────────────────────

def _flat_urdf(src: str, package_root: str) -> str:
    with open(src) as f:
        content = f.read()
    abs_root = os.path.abspath(package_root)
    content  = content.replace("package://ur10e_description/", f"{abs_root}/")
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG DRAWING — seam lines in the viewport
# ─────────────────────────────────────────────────────────────────────────────

def draw_seam_lines(seam_edges):
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        pts0 = [e[0].tolist() for e in seam_edges]
        pts1 = [e[1].tolist() for e in seam_edges]
        cols = [(1.0, 0.25, 0.1, 1.0)] * len(seam_edges)
        sizes = [3.5] * len(seam_edges)
        draw.draw_lines(pts0, pts1, cols, sizes)
    except Exception as exc:
        carb.log_warn(f"Debug draw unavailable: {exc}")


def draw_tcp_point(pos):
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_points([pos.tolist()], [(0.1, 1.0, 0.1, 1.0)], [8.0])
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  WORKPIECE VISUAL — add a translucent shape prim so the user can see it
# ─────────────────────────────────────────────────────────────────────────────

def add_workpiece(world: World, shape: str, size: float):
    if shape in ("cube", "box"):
        lx = size
        ly = size if shape == "cube" else size * 0.6
        lz = size if shape == "cube" else size * 0.8
        center = np.array([_REACH_X, 0.0, _TABLE_Z + lz / 2])
        world.scene.add(VisualCuboid(
            prim_path="/World/workpiece",
            name="workpiece",
            position=center,
            scale=np.array([lx, ly, lz]),
            color=np.array([0.25, 0.55, 0.85]),
        ))
    # For l_shape / triangle_prism a VisualCuboid bounding box is shown as a
    # rough guide — actual seam lines drawn via debug_draw are authoritative.
    else:
        center = np.array([_REACH_X, 0.0, _TABLE_Z + size / 2])
        world.scene.add(VisualCuboid(
            prim_path="/World/workpiece",
            name="workpiece",
            position=center,
            scale=np.array([size, size, size]),
            color=np.array([0.25, 0.55, 0.85]),
        ))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n[seam_tracker] shape={args.shape}  size={args.size}m  "
          f"density={args.density}pts/edge  speed={args.speed}f/pt\n")

    # ── Scene ─────────────────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Extra lighting
    stage = omni.usd.get_context().get_stage()
    dl = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
    dl.CreateIntensityAttr(800)

    viewports.set_camera_view(eye=[1.8, 1.4, 1.2], target=[0.5, 0.0, 0.3])

    # ── Load UR10e from URDF ──────────────────────────────────────────────────
    flat_urdf = _flat_urdf(_URDF_SRC, _MESH_ROOT)

    _, import_cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    import_cfg.merge_fixed_joints    = False
    import_cfg.convex_decomp         = False
    import_cfg.import_inertia_tensor = True
    import_cfg.fix_base              = True
    import_cfg.distance_scale        = 1.0

    ok, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=flat_urdf,
        import_config=import_cfg,
        dest_path=_ROBOT_PRIM,
    )
    os.unlink(flat_urdf)

    if not ok:
        carb.log_error("URDF import failed — aborting.")
        simulation_app.close()
        sys.exit(1)

    # ── Wrap articulation and wire up IK ──────────────────────────────────────
    ur10e = Articulation(prim_path=_ROBOT_PRIM)
    world.scene.add(ur10e)

    # ── Workpiece visual ──────────────────────────────────────────────────────
    add_workpiece(world, args.shape, args.size)

    # ── Reset world (initialises physics) ────────────────────────────────────
    world.reset()
    simulation_app.update()

    # ── Lula IK solver ────────────────────────────────────────────────────────
    # Uses Isaac Sim's built-in kinematics — no MoveIt required.
    lula = LulaKinematicsSolver(
        robot_description_path=_ROBOT_DESC,
        urdf_path=_URDF_SRC,   # Lula only reads joint topology, not meshes
    )
    ik_solver   = ArticulationKinematicsSolver(ur10e, lula, _EE_FRAME)
    art_ctrl    = ur10e.get_articulation_controller()

    # ── Seam path ─────────────────────────────────────────────────────────────
    seam_edges = SHAPES[args.shape](args.size)
    waypoints  = seam_to_waypoints(seam_edges, args.density, args.height)

    print(f"[seam_tracker] {len(seam_edges)} seam edges  "
          f"→  {len(waypoints)} waypoints total")

    # Draw seam lines once (they persist in the viewport)
    world.step(render=True)
    draw_seam_lines(seam_edges)

    # ── Simulation loop ───────────────────────────────────────────────────────
    wp_idx         = 0
    frame_at_wp    = 0
    total_wp       = len(waypoints)
    ik_fail_streak = 0
    loops_done     = 0

    print("[seam_tracker] Starting seam tracking …  (Ctrl-C or close window to stop)")

    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            continue

        # Reset articulation on the first step
        if world.current_time_step_index == 0:
            world.reset()
            wp_idx = 0
            frame_at_wp = 0

        if wp_idx >= total_wp:
            loops_done += 1
            print(f"[seam_tracker] Loop {loops_done} complete — restarting seam.")
            wp_idx = 0
            frame_at_wp = 0
            # Redraw seam lines (they clear on reset)
            draw_seam_lines(seam_edges)
            continue

        target_pos, target_quat = waypoints[wp_idx]

        # Solve IK and apply
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat,
        )

        if success:
            art_ctrl.apply_action(action)
            draw_tcp_point(target_pos)
            ik_fail_streak = 0
        else:
            ik_fail_streak += 1
            if ik_fail_streak <= 3:
                carb.log_warn(
                    f"[seam_tracker] IK failed at waypoint {wp_idx} "
                    f"pos={np.round(target_pos,3)} — skipping."
                )

        # Advance waypoint every N frames so the robot has time to move
        frame_at_wp += 1
        if frame_at_wp >= args.speed:
            wp_idx      += 1
            frame_at_wp  = 0

    world.stop()
    simulation_app.close()
    print("[seam_tracker] Done.")


if __name__ == "__main__":
    main()
