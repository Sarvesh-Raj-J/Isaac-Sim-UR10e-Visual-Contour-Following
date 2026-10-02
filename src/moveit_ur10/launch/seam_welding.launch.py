"""
seam_welding.launch.py
======================
All-in-one launch for UR10e seam welding with MoveIt2 + IsaacSim.

Start positions match the live robot echo:
  shoulder_pan: -1.5007  shoulder_lift: -1.8652  elbow: -0.8619
  wrist_1:       4.2768  wrist_2:        1.6200  wrist_3: -3.0813

Flow after bringup (driven by seam_weld_supervisor):
  1. Move to OVERVIEW pose  → detector shows wide image
  2. Move to SCAN pose      → detector triggered, publishes /seam_edges
  3. OMPL approach + Cartesian traverse for each seam edge

Requires IsaacSim running and publishing /rgb, /depth, /camera_info.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── MoveIt config ──────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10")
        .robot_description(file_path="config/ur10e_robot.urdf.xacro")
        .robot_description_semantic(file_path="config/ur10e_robot.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .parameter("use_sim_time", True)
        .to_moveit_configs()
    )

    ros2_controllers_path = str(
        moveit_config.package_path / "config/ros2_controllers.yaml"
    )
    rviz_config_file = str(
        moveit_config.package_path / "config/moveit.rviz"
    )

    # Wide-scan / home joint positions (the overview pose from /joint_states echo)
    initial_joint_positions = {
        "initial_positions": {
            "shoulder_pan_joint":  -1.5007,
            "shoulder_lift_joint": -1.8652,
            "elbow_joint":         -0.8619,
            "wrist_1_joint":        4.2768,
            "wrist_2_joint":        1.6200,
            "wrist_3_joint":       -3.0813,
        }
    }

    # ── Core infrastructure ────────────────────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        parameters=[{"use_sim_time": True}],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            initial_joint_positions,
            {"use_sim_time": True},
        ],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            initial_joint_positions,
            {"use_sim_time": True},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True},
        ],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            ros2_controllers_path,
            {"use_sim_time": True},
        ],
    )

    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager",
                   "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    # Sync IsaacSim to the home (wide-scan) joint positions via arm_controller
    initial_pose_cmd = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            "/arm_controller/joint_trajectory",
            "trajectory_msgs/msg/JointTrajectory",
            (
                "{"
                "joint_names: ["
                "  shoulder_pan_joint, shoulder_lift_joint, elbow_joint,"
                "  wrist_1_joint, wrist_2_joint, wrist_3_joint"
                "],"
                "points: [{"
                "  positions: [-1.5007, -1.8652, -0.8619, 4.2768, 1.6200, -3.0813],"
                "  velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],"
                "  time_from_start: {sec: 4, nanosec: 0}"
                "}]"
                "}"
            ),
        ],
        output="screen",
    )

    # ── Seam welding pipeline nodes ────────────────────────────────────────────
    seam_detector_node = Node(
        package="moveit_ur10",
        executable="seam_weld_detector.py",
        name="seam_weld_detector",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    seam_supervisor_node = Node(
        package="moveit_ur10",
        executable="seam_weld_supervisor.py",
        name="seam_weld_supervisor",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"planning_group": "arm"},
            {"ee_link": "tool0"},
            {"tool_offset": 0.05},
            {"approach_lift": 0.12},
            {"cart_min_fraction": 0.80},
        ],
    )

    # After arm_controller is spawned:
    #   t=0  → send initial pose + launch RViz
    #   t=8  → launch detector + supervisor (controllers must be ready)
    delay_after_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_spawner,
            on_exit=[
                TimerAction(period=2.0, actions=[initial_pose_cmd]),
                rviz,
                TimerAction(
                    period=8.0,
                    actions=[seam_detector_node, seam_supervisor_node],
                ),
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        static_tf,
        robot_state_publisher,
        move_group,
        ros2_control_node,
        jsb_spawner,
        arm_spawner,
        delay_after_arm,
    ])
