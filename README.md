# ecn_ga-mppi
ECN M1 Group Project: Integrating MPPI with Genetic Algorithms

## Unicycle Model and Random Path Generator

The `src` folder includes two Python files:

1. **`motionModel.py`**:  
    This file defines the unicycle motion model for the TurtleBot and provides a path prediction function, `predict`, within the `turtleModel` class.

2. **`utils.py`**:  
    This script performs the following tasks:  
    - Generates a reference path (a randomly generated line of length `horizon`).  
    - Creates `num_paths` random paths using the `turtleModel` class.
    - Identifies the closest path to the reference using the Euclidean norm.
    - Identifies the best path acc. to control law.
    - Visualizes all random paths and highlights the closest path, best control path and the reference path, using `matplotlib`.

5. **`ga_costs.py`**:
    This script contains three variations of cost functions for control law cost with three open functions: `compute_path_curvature_cost`, `compute_control_variation_cost`, and `compute_control_effort_cost`. Lastly, `compute_total_cost` function which computes all costs with user-defined weights.

6. **`MPPI_GA.py`**:
    Implements GA using a population of U (inputs), performs elitism, cross-over and mutation based on tunable parameters, and spits out a best_U and best_X (best GA path), with hard-coded tresholds for time, cost convergance and no. of generations.

7. **`mppi-ga/mppi-node.py`**:
    Contains the ROS integration for the MPPI_GA class. It creates a node called `mppi_node`, which subscribes to /transformed_global_plan (temp. implementation by utilizing the transformed global path from DWB, or any other controller) and /goal_pose. It then runs an instance of MPPI_GA with the unicyle motion model and publishes /cmd_vel_mppi (a custom command topic) and /local_plan.

8. **`turtlebot3_ws`**:
    This contains a modified implementation of the turtlebot3 packages by [ROBOTIS](https://github.com/ROBOTIS-GIT) with slightly modified params in gazebo and rviz. Use this turtlebot3 workspace for the project demonstration.

### Usage
Set these params in your ~/.bashrc file first:
```bash
# ROS2 environment variables
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export TURTLEBOT3_MODEL=waffle
export ROS_DOMAIN_ID=31
```
Make sure you've installed cyclonedds, otherwise this will throw an error!

Then, in `turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf`, in the <ros> tag, add `<remapping>cmd_vel:=cmd_vel_mppi</remapping>`, such that it looks like:
```xml
    <plugin name="turtlebot3_diff_drive_echo" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <remapping>cmd_vel:=cmd_vel_mppi</remapping>
      </ros>
      
      <update_rate>30</update_rate>

      <!-- wheels -->
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      .....
```
So that the turtlebot3_navigation doesn't control our robot anymore.

To run the simulations, clone this repo, build it and source it, then in your workspace root, run:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
To launch the default turtlebot3 world.
```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True
```
To launch the nav2 with some modified params in humble/waffle.yaml. And finally, to run our implementation of MPPI with GA, run:
```bash
ros2 run unicycle mppi-node.py
```

### Results
Here's a demonstration of the MPPI-GA:
[MPPI-GA test video](https://drive.google.com/file/d/1QLVFEKqPLOLiY7PeP1puDP1UkbbS4yys/view?usp=sharing)