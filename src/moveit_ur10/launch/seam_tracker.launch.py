"""
seam_tracker.launch.py
=======================
All-in-one launch for UR10e seam tracking with MoveIt2 + RViz.

Nodes started:
  1. Robot State Publisher
  2. MoveGroup  (OMPL planning pipeline)
  3. RViz2
  4. Static TF  (world → base_link)
  5. ros2_control_node
  6. joint_state_broadcaster
  7. arm_controller
  8. seam_detector   – CV edge detector (publishes /seam_edges)
  9. seam_tracker    – MoveIt trajectory executor (subscribes /seam_edges)

Isaac Sim must already be running and publishing /rgb, /depth, /camera_info.

Example:
  ros2 launch moveit_ur10 seam_tracker.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction, ExecuteProcess
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── MoveIt config (includes OMPL pipeline by default) ─────────────────────
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

    # ── Initial joint positions ────────────────────────────────────────────────
    initial_joint_positions = {
        "initial_positions": {
            "shoulder_pan_joint": 1.3531,
            "shoulder_lift_joint": -1.2291,
            "elbow_joint": 1.0821,
            "wrist_1_joint": 4.9025,
            "wrist_2_joint": -1.7125,
            "wrist_3_joint": 2.8561,
        }
    }

    # ── 1. Robot State Publisher ───────────────────────────────────────────────
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

    # ── 2. Static TF: world → base_link ───────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. MoveGroup (OMPL pipeline) ──────────────────────────────────────────
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

    # ── 4. RViz ───────────────────────────────────────────────────────────────
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

    # ── 5. ros2_control_node ──────────────────────────────────────────────────
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

    # ── 6. joint_state_broadcaster ────────────────────────────────────────────
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager",
                   "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    # ── 7. arm_controller ─────────────────────────────────────────────────────
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    # ── 8. Send initial joint positions once arm_controller is ready ─────────────
    # TopicBasedSystem reads /joint_states from Isaac Sim, so the only way to
    # command the robot to a specific start pose is to publish a trajectory.
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
                "  positions: [1.3531, -1.2291, 1.0821, 4.9025, -1.7125, 2.8561],"
                "  velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],"
                "  time_from_start: {sec: 5, nanosec: 0}"
                "}]"
                "}"
            ),
        ],
        output="screen",
    )

    # After arm_controller exits the spawn phase → send initial pose, then RViz
    delay_after_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_spawner,
            on_exit=[
                TimerAction(period=2.0, actions=[initial_pose_cmd]),
                rviz,
            ],
        )
    )

    return LaunchDescription([
        # Core infrastructure
        static_tf,
        robot_state_publisher,
        move_group,
        ros2_control_node,
        jsb_spawner,
        arm_spawner,
        # Deferred: initial pose command + RViz after controllers ready
        delay_after_arm,
    ])
