from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. Setup Launch Configuration for consistency
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 2. Build the Configuration
    moveit_config = (
        MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10")
        .robot_description(file_path="config/ur10e_robot.urdf.xacro")
        .robot_description_semantic(file_path="config/ur10e_robot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .parameter("use_sim_time", True) # Injects into MoveIt internal logic
        .to_moveit_configs()
    )

    ros2_controllers_path = str(moveit_config.package_path / "config/ros2_controllers.yaml")

    # A. The "Brain" (Move Group)
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True} # <--- REQUIRED
        ],
    )

    # B. The "Visualizer" (RViz)
    rviz_config_file = str(moveit_config.package_path / "config/moveit.rviz")
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
            {"use_sim_time": True} # <--- REQUIRED
        ],
    )

    # C. The "Bridge" (ROS 2 Control Node)
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            ros2_controllers_path,
            {"use_sim_time": True} # <--- REQUIRED
        ],
        output="screen",
    )

    # D. The "Transform Publisher" (Robot State Publisher)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": True} # <--- REQUIRED
        ],
    )

    # E. Spawn the Controllers (Spawners also need sim_time)
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}]
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": True}]
    )

    # F. Sequencing
    delay_rviz_after_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[run_rviz_node],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        ros2_control_node,
        robot_state_publisher,
        run_move_group_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        delay_rviz_after_spawner, 
    ])