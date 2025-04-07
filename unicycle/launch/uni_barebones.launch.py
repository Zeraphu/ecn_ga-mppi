import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare('unicycle').find('unicycle')
    default_model_path = os.path.join(pkg_share, 'urdf', 'uni_barebones.xacro')

    rsp_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': Command(['xacro ', LaunchConfiguration('model')])
        }],
    )

    jsp_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        arguments=[LaunchConfiguration('model')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),}],
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'uni_barebones.rviz')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),}],
    )

    spawn_entity = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'unicycle', '-topic', 'robot_description'],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),}],
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'model',
            default_value=default_model_path,
            description='Absolute path to robot urdf file'),
        launch.actions.DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo) clock if true'),
        launch.actions.ExecuteProcess(cmd=['gazebo', '--verbose', 
                                           '-s', 'libgazebo_ros_init.so', 
                                           '-s', 'libgazebo_ros_factory.so'], 
                                           output='screen'),
        spawn_entity,
        rsp_node,
        jsp_node,
        rviz_node,
    ])
