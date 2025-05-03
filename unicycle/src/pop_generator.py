#!/usr/bin/env python3

from motionModel import turtleModel
import numpy as np
import matplotlib.pyplot as plt
import time

turtle = turtleModel()
horizon = 100 # Number of steps

def generate_reference_path():
    """
    Generates a reference path using the turtle model.
    :return: Reference path.
    """
    turtle.set_state(0, 0, 0)
    turtle.generate_input(horizon)
    return turtle.predict(horizon)

def generate_population(num_paths:int = 100):
    """
    Generates a population of paths using the turtle model.
    :param num_paths: Number of paths to generate.
    :return: List of paths.
    """
    paths = []
    for _ in range(num_paths):
        turtle.set_state(0, 0, 0)
        paths.append(turtle.predict(horizon))
    return paths

def closest_path(ref_path, paths):
    """
    Finds the closest path to the reference path.
    :param ref_path: Reference path.
    :param paths: List of paths.
    :return: Closest path.
    """
    min_distance = float('inf')
    closest_path = None
    for path in paths:
        distance = np.linalg.norm(ref_path - path)
        if distance < min_distance:
            min_distance = distance
            closest_path = path
    return closest_path

def visualize(num_paths:int = 100):
    """
    Visualizes the generated paths.
    """
    start = time.time()
    ref_path = generate_reference_path()
    paths = generate_population(num_paths)
    closest = closest_path(ref_path, paths)
    end = time.time()
    print(f"\nTime taken to generate and visualize paths: {end - start:.4f} seconds\n")
    
    # Plot the path population, reference path and the closest path
    for path in paths:
        plt.plot(path[:, 0], path[:, 1])
    plt.plot(ref_path[:, 0], ref_path[:, 1], color='black', label='Reference Path', linewidth=5)
    plt.plot(closest[:, 0], closest[:, 1], color='red', label='Closest Path', linewidth=5)
    plt.legend()
    plt.title("Generated Paths")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    visualize(1000)