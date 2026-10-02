from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("ur10e_robot", package_name="moveit_ur10")
        .robot_description(file_path="config/ur10e_robot.urdf.xacro")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        # This line injects use_sim_time: True into all MoveIt nodes
        .parameter("use_sim_time", True) 
        .to_moveit_configs()
    )
    
    return generate_demo_launch(moveit_config)