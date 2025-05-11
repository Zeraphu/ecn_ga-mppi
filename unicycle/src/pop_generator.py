#!/usr/bin/env python3

from motionModel import turtleModel
import numpy as np
import matplotlib.pyplot as plt
import time
from ga_costs import compute_total_cost
from ga_costs import compute_ctr_cost, compute_euc_cost

turtle = turtleModel()
horizon, num_paths = 100, 100

<<<<<<< HEAD
def generate_reference_path(x:float = 0.0, y:float = 0.0, theta:float = 0.0):
=======
def generate_reference_path(horizon:int = 100, 
                            x:float = 0.0, y:float = 0.0, theta:float = 0.0):
>>>>>>> 2717ca8 (WIP: gen_algo initial push.)
    """
    Generates a reference path using the turtle model.
    :return: Reference path.
    """
<<<<<<< HEAD
    turtle.set_state(x, y, theta)
    return turtle.predict(horizon)

def generate_population(num_paths:int = 100, x:float = 0.0, y:float = 0.0, theta:float = 0.0):
=======
    turtle.reset_state(x, y, theta)
    turtle.generate_input(horizon)
    return turtle.predict(horizon)

def generate_population(horizon:int = 100, num_paths:int = 100, 
                        x:float = 0.0, y:float = 0.0, theta:float = 0.0):
>>>>>>> 2717ca8 (WIP: gen_algo initial push.)
    """
    Generates a population of paths using the turtle model.
    :param num_paths: Number of paths to generate.
    :return: List of paths.
    """
    inputs, paths = [], []
    for _ in range(num_paths):
<<<<<<< HEAD
        turtle.set_state(x, y, theta)
=======
        turtle.reset_state(x, y, theta)
        turtle.generate_input(horizon)
>>>>>>> 2717ca8 (WIP: gen_algo initial push.)
        input, path = turtle.predict(horizon)
        paths.append(path)
        inputs.append(input)
    return inputs, paths

def closest_path(ref_path, paths):
    """
    Finds the closest path to the reference path based on the cost function.
    NOTE: param paths has to be a list of paths!
    :param ref_path: Reference path.
    :param paths: List of paths.
    :return: minimum cost and closest path.
    """
<<<<<<< HEAD
    if len(ref_path) != len(paths[0]):
        raise ValueError("Reference path and paths must have the same length.")
    
    min_distance = float('inf')
    closest_path = None
    for path in paths:
        distance = np.linalg.norm(ref_path - path)
        if distance < min_distance:
            min_distance = distance
            closest_path = path
    return min_distance, closest_path

def best_ctr_path(inputs, paths):
    """
    Finds the best control path based on the cost function.
    :param inputs: List of inputs.
    :param paths: List of paths.
    :return: Best control path.
    """
    min_cost = float('inf')
    best_path = None
    for input, path in zip(inputs, paths):
        cost = compute_total_cost(path, input)
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return min_cost, best_path
=======
    min_cost = float('inf')
    closest_path = None
>>>>>>> 2717ca8 (WIP: gen_algo initial push.)

    for path in paths:
        cost = compute_euc_cost(path, ref_path)
        if cost < min_cost:
            min_cost = cost
            closest_path = path
    return min_cost, closest_path

def best_ctr_path(inputs, paths):
    """
    Finds the best control path based on the cost function.
    :param inputs: List of inputs.
    :param paths: List of paths.
    :return: Best control path.
    """
<<<<<<< HEAD
    start = time.time()

    _, ref_path = generate_reference_path()
    inputs, paths = generate_population(num_paths)
    _, closest = closest_path(ref_path, paths)
    _, ctr_closest = best_ctr_path(inputs, paths)

    end = time.time()
    print(f"\nTime taken to generate and visualize paths: {end - start:.4f} seconds\n")
    
    # Plot the path population, reference path and the closest path
    for path in paths:
        plt.plot(path[:, 0], path[:, 1])

    plt.plot(ref_path[:, 0], ref_path[:, 1], color='black', label='Reference Path', linewidth=5)
    plt.plot(closest[:, 0], closest[:, 1], color='red', label='Closest Path', linewidth=5)
    plt.plot(ctr_closest[:, 0], ctr_closest[:, 1], color='blue', label='Best Control Path', linewidth=5)
=======
    min_cost = float('inf')
    best_path = None
    for input, path in zip(inputs, paths):
        cost = compute_ctr_cost(path, input)
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return min_cost, best_path

def visualize(paths = [], ref_path = [], closest = [], ctr_closest = []):
    """
    Visualizes the generated paths using matplotlib.
    :param paths: List of paths.
    :param ref_path: Reference path.
    :param closest: Closest path to the reference path.
    :param ctr_closest: Best control path.
    NOTE: Needs the rest of the path types to be manually added.
    """
    if len(paths) == 0:
        raise ValueError("\nNo paths to visualize.\n")
        return
    for path in paths:
        plt.plot(path[:, 0], path[:, 1])

    if len(ref_path) == 0:
        raise ValueError("\nNo reference path to visualize.\n")
        return
    else: plt.plot(ref_path[:, 0], ref_path[:, 1], 
                 color='black', label='Reference Path', linewidth=5)
    
    if len(closest) == 0:
        print("\nNo closest path to visualize.\n")
    else: plt.plot(closest[:, 0], closest[:, 1], 
                   color='red', label='Closest Path', linewidth=5)
    
    if len(ctr_closest) == 0:
        print("\nNo best control path to visualize.\n")
    else: plt.plot(ctr_closest[:, 0], ctr_closest[:, 1], 
                   color='blue', label='Best Control Path', linewidth=5)
>>>>>>> 2717ca8 (WIP: gen_algo initial push.)

    plt.legend()
    plt.title("Generated Paths")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    start = time.time()

    _, ref_path = generate_reference_path(horizon=horizon)
    inputs, paths = generate_population(horizon=horizon, num_paths=num_paths)
    _, closest = closest_path(ref_path, paths)
    _, ctr_closest = best_ctr_path(inputs, paths)

    end = time.time()
    print(f"\nTime taken to generate and visualize paths: {end - start:.4f} seconds\n")
    
    visualize(paths=paths,
             ref_path=ref_path,
             closest=closest,
             ctr_closest=ctr_closest)