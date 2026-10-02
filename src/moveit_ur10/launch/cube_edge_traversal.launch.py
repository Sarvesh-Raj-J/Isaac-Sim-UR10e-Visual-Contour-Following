"""
cube_edge_traversal.launch.py
==============================
All-in-one launch for UR10e cube edge traversal with MoveIt2 + IsaacSim.

Flow (driven by cube_edge_supervisor):
  1. Move to HOME pose → camera sees cube
  2. Trigger /detect_cube → cube_edge_detector publishes /cube_edges (12 edges)
  3. For each edge: OMPL approach + Cartesian traverse along the edge
  4. Return HOME when done

Requires IsaacSim publishing /rgb, /depth, /camera_info.
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

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║      WORKSPACE WALLS — edit here to move any wall (metres)         ║
    # ║   Values match Position X / Y / Z shown in RViz → Scene Objects   ║
    # ╠════════════════╦══════════╦══════════╦══════════╗                  ║
    # ║  Wall          ║    X     ║    Y     ║    Z     ║                  ║
    # ╠════════════════╬══════════╬══════════╬══════════╣                  ║
    WALL_LEFT    = ( -0.81,    0.95,    0.725 )  # left side
    WALL_RIGHT   = (  0.81,    0.95,    0.725 )  # right side
    WALL_BACK    = (  0.00,    0.09,    0.725 )  # behind robot
    WALL_FRONT   = (  0.00,    1.81,    0.725 )  # in front of robot
    WALL_FLOOR   = (  0.00,    0.95,   -0.06  )  # below
    WALL_CEILING = (  0.00,    0.95,    1.51  )  # above
    # ╚════════════════╩══════════╩══════════╩══════════╝

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

    # Home joint positions (same as seam_welding — wide-scan pose)
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

    # Sync robot to home joints on startup
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

    # ── Cube traversal pipeline nodes ──────────────────────────────────────────
    cube_detector_node = Node(
        package="moveit_ur10",
        executable="cube_edge_detector.py",
        name="cube_edge_detector",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    cube_supervisor_node = Node(
        package="moveit_ur10",
        executable="cube_edge_supervisor.py",
        name="cube_edge_supervisor",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"planning_group":    "arm"},
            {"ee_link":           "tool0"},
            {"tool_offset":       0.02},
            {"approach_lift":     0.10},
            {"cart_min_fraction": 0.80},
            # wall centre positions — driven by the tuples at the top of this file
            {"wall_left_x":    WALL_LEFT[0]},    {"wall_left_y":    WALL_LEFT[1]},    {"wall_left_z":    WALL_LEFT[2]},
            {"wall_right_x":   WALL_RIGHT[0]},   {"wall_right_y":   WALL_RIGHT[1]},   {"wall_right_z":   WALL_RIGHT[2]},
            {"wall_back_x":    WALL_BACK[0]},    {"wall_back_y":    WALL_BACK[1]},    {"wall_back_z":    WALL_BACK[2]},
            {"wall_front_x":   WALL_FRONT[0]},   {"wall_front_y":   WALL_FRONT[1]},   {"wall_front_z":   WALL_FRONT[2]},
            {"wall_floor_x":   WALL_FLOOR[0]},   {"wall_floor_y":   WALL_FLOOR[1]},   {"wall_floor_z":   WALL_FLOOR[2]},
            {"wall_ceiling_x": WALL_CEILING[0]}, {"wall_ceiling_y": WALL_CEILING[1]}, {"wall_ceiling_z": WALL_CEILING[2]},
        ],
    )

    # Sequencing: initial pose + RViz after arm_controller, then pipeline nodes
    delay_after_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_spawner,
            on_exit=[
                TimerAction(period=2.0, actions=[initial_pose_cmd]),
                rviz,
                TimerAction(
                    period=8.0,
                    actions=[cube_detector_node, cube_supervisor_node],
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
