#!usr/bin/env python3
from motionModel import underwaterRobotModel
import numpy as np

rov_model = underwaterRobotModel()

def generate_population(num_samples: int = 100, horizon: int = 100):
    pop = []
    for _ in range(num_samples):
        # Initialize robot with default parameters
        rov_model = underwaterRobotModel(dt=0.1)
        rov_model.set_state(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        rov_model.generate_input(horizon=horizon)
        pop.append(rov_model.predict())
    return pop

def generate_reference(horizon: int = 100,):
    ref = []
    # Initialize robot with default parameters
    rov_model = underwaterRobotModel(dt=0.1)
    rov_model.set_state(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    rov_model.generate_input(horizon=horizon)
    ref = rov_model.predict()
    return ref

def closest_path_to_ref(trajectory, ref):
    """
    Find the closest path to the reference trajectory
    :param trajectory: The trajectory to compare
    :param ref: The reference trajectory
    :return: The closest path to the reference
    """
    min_distance = float('inf')
    closest_path = None

    for path in trajectory:
        distance = np.linalg.norm(path - ref)
        if distance < min_distance:
            min_distance = distance
            closest_path = path
    return closest_path

def visualize_population(population, ref, closest_path):
    """
    Visualize the generated population of trajectories, while making reference path and closest path highlighted using thicker lines.
    :param population: The population of trajectories
    :param ref: The reference trajectory
    :param closest_path: The closest path to the reference
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the population of trajectories
    for trajectory in population:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.5)

    # Plot the reference trajectory
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color='red', linewidth=2, label='Reference Path')

    # Plot the closest path to the reference
    ax.plot(closest_path[:, 0], closest_path[:, 1], closest_path[:, 2], color='green', linewidth=4, label='Closest Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate a population of trajectories
    population = generate_population(num_samples=100, horizon=100)
    
    # Generate a reference trajectory
    ref = generate_reference(horizon=100)
    
    # Find the closest path to the reference
    closest_path = closest_path_to_ref(population, ref)
    
    # Visualize the population, reference, and closest path
    visualize_population(population, ref, closest_path)