"""
Isaac Sim standalone script — UR10e + MoveIt/RViz bridge
=========================================================
Run this BEFORE the ROS2 launch file.

What it does:
  - Loads the UR10e from the local URDF
  - Subscribes to /joint_command  (published by topic_based_ros2_control)
    and drives the Isaac Sim articulation every frame
  - Publishes /joint_states       (read by topic_based_ros2_control)
  - Publishes /clock              (required because use_sim_time: true)

Then in a separate terminal run:
  ros2 launch moveit_ur10 move_group.launch.py
"""

import sys
import os
import tempfile

# ── Bootstrap Isaac Sim ──────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import carb
import omni.kit.commands
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, viewports, stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdPhysics, Sdf, UsdLux

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE_SRC = "/home/robotics/seam_ws/src"
URDF_SRC      = f"{WORKSPACE_SRC}/ur10e_description/urdf/ur10e.urdf"
MESHES_SRC    = f"{WORKSPACE_SRC}/ur10e_description"
ROBOT_PRIM    = "/ur10e_robot"

# ── Enable ROS2 bridge ───────────────────────────────────────────────────────
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ── Patch package:// URIs so Isaac Sim can find meshes without ROS ───────────
def _make_flat_urdf(src_urdf: str, package_root: str) -> str:
    with open(src_urdf) as f:
        content = f.read()
    abs_root = os.path.abspath(package_root)
    content = content.replace("package://ur10e_description/", f"{abs_root}/")
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return tmp.name

flat_urdf = _make_flat_urdf(URDF_SRC, MESHES_SRC)

# ── Import URDF ──────────────────────────────────────────────────────────────
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints  = False
import_config.convex_decomp       = False
import_config.import_inertia_tensor = True
import_config.fix_base            = True
import_config.distance_scale      = 1.0
import_config.make_default_prim   = True

status, robot_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=flat_urdf,
    import_config=import_config,
    dest_path=ROBOT_PRIM,
)
os.unlink(flat_urdf)

if not status:
    carb.log_error("URDF import failed — check the path and meshes")
    simulation_app.close()
    sys.exit(1)

carb.log_info(f"UR10e imported at prim path: {robot_prim_path}")

# ── Scene setup ───────────────────────────────────────────────────────────────
simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)
viewports.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.0, 0.5])

# Ground plane
omni.kit.commands.execute(
    "CreateMeshPrimWithDefaultXform",
    prim_type="Plane",
    prim_path="/World/GroundPlane",
)

# Lighting
usd_stage = omni.usd.get_context().get_stage()
dist_light = UsdLux.DistantLight.Define(usd_stage, Sdf.Path("/World/DistantLight"))
dist_light.CreateIntensityAttr(500)

simulation_app.update()

# ── Build the Action Graph ────────────────────────────────────────────────────
#
#   OnImpulseEvent (tick every frame)
#     ├─► PublishClock          → /clock
#     ├─► PublishJointState     → /joint_states   (robot → ROS)
#     ├─► SubscribeJointState   ← /joint_command  (ROS → robot)
#     └─► ArticulationController ← SubscribeJointState outputs
#
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent",       "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime",          "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context",          "isaacsim.ros2.bridge.ROS2Context"),
                # Clock
                ("PublishClock",         "isaacsim.ros2.bridge.ROS2PublishClock"),
                # Joint state feedback to ROS (→ /joint_states)
                ("PublishJointState",    "isaacsim.ros2.bridge.ROS2PublishJointState"),
                # Receive joint commands from ROS (← /joint_command)
                ("SubscribeJointState",  "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                # Apply received commands to the robot
                ("ArticulationCtrl",     "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                # Tick everything together
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationCtrl.inputs:execIn"),
                # ROS2 context
                ("ROS2Context.outputs:context",    "PublishClock.inputs:context"),
                ("ROS2Context.outputs:context",    "PublishJointState.inputs:context"),
                ("ROS2Context.outputs:context",    "SubscribeJointState.inputs:context"),
                # Timestamps
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                # Wire subscriber → articulation controller
                ("SubscribeJointState.outputs:jointNames",      "ArticulationCtrl.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationCtrl.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationCtrl.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",   "ArticulationCtrl.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                # Point ArticulationController at our robot prim
                ("ArticulationCtrl.inputs:robotPath",          ROBOT_PRIM),
                # /joint_states — what topic_based_ros2_control reads as feedback
                ("PublishJointState.inputs:topicName",         "joint_states"),
                ("PublishJointState.inputs:targetPrim",        [usdrt.Sdf.Path(ROBOT_PRIM)]),
                # /joint_command — what topic_based_ros2_control publishes as commands
                ("SubscribeJointState.inputs:topicName",       "joint_command"),
            ],
        },
    )
    carb.log_info("Action Graph created successfully")
except Exception as exc:
    carb.log_error(f"Action Graph creation failed: {exc}")
    simulation_app.close()
    sys.exit(1)

# ── Run ───────────────────────────────────────────────────────────────────────
simulation_app.update()
simulation_context.initialize_physics()
simulation_context.play()

print("\n" + "="*60)
print("Isaac Sim UR10e ready.")
print("  /joint_command  → robot articulation (from MoveIt)")
print("  /joint_states   → ROS2 control feedback")
print("  /clock          → sim time published")
print("Now launch in another terminal:")
print("  cd ~/seam_ws && ros2 launch moveit_ur10 move_group.launch.py")
print("="*60 + "\n")

while simulation_app.is_running():
    simulation_context.step(render=True)
    # Tick the action graph every frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"),
        True,
    )

simulation_context.stop()
simulation_app.close()
