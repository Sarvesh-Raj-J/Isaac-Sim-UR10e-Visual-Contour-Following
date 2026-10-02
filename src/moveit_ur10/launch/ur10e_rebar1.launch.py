from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # --- INITIAL JOINT POSITIONS (Scan Pose) ---
    initial_joint_positions = {
        "initial_positions": {
            "shoulder_pan_joint": 0.7508,
            "shoulder_lift_joint": -1.7514,
            "elbow_joint": 1.8618,
            "wrist_1_joint": -0.1103,
            "wrist_2_joint": 0.8394,
            "wrist_3_joint": 0.0007,
        }
    }

    moveit_config = (
        MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10")
        .robot_description(file_path="config/ur10e_robot.urdf.xacro")
        .robot_description_semantic(file_path="config/ur10e_robot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .parameter("use_sim_time", True)
        .to_moveit_configs()
    )

    ros2_controllers_path = str(moveit_config.package_path / "config/ros2_controllers.yaml")
    rviz_config_file = str(moveit_config.package_path / "config/moveit.rviz")

    # --- Core MoveIt / ros2_control nodes ---
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(), 
            initial_joint_positions, # Injected
            {"use_sim_time": True}
        ],
    )

    run_rviz_node = Node(
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
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            ros2_controllers_path,
            {"use_sim_time": True},
        ],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description, 
            initial_joint_positions, # Injected
            {"use_sim_time": True}
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}],
    )

    delay_rviz_after_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[run_rviz_node],
        )
    )

    # --- Rebar Pipeline Nodes ---
    rebar_detector_node = Node(
        package="moveit_ur10",
        executable="rebar_detector.py",
        name="rebar_detector",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"rebar_plane_y": 1.0},
        ],
    )

    rebar_supervisor_node = Node(
        package="moveit_ur10",
        executable="rebar_supervisor.py",
        name="rebar_supervisor",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"grid_rows": 2},
            {"grid_cols": 2},
            {"hover_time": 3.0},
            {"approach_distance": 0.15}, # Updated for better planning clearance
            {"planning_group": "arm"},
            {"ee_link": "tool0"},
            {"controller_action": "/arm_controller/follow_joint_trajectory"},
        ],
    )

    delay_rebar_nodes = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[
                TimerAction(period=6.0, actions=[rebar_detector_node, rebar_supervisor_node])
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        ros2_control_node,
        robot_state_publisher,
        run_move_group_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        delay_rviz_after_spawner,
        delay_rebar_nodes,
    ])
