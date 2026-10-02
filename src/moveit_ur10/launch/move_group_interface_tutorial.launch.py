from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10").to_moveit_configs()

    move_group_interface_tutorial = Node(
        name="move_group_interface_tutorial",
        package="moveit_ur10",
        executable="move_group_interface_tutorial",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([move_group_interface_tutorial])
