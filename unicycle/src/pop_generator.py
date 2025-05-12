#!/usr/bin/env python3

from motionModel import turtleModel
import numpy as np
import matplotlib.pyplot as plt
import time
from ga_costs import compute_ctr_cost, compute_euc_cost, compute_advance

turtle = turtleModel()
horizon, num_paths = 100, 100

def generate_reference_path(horizon:int = 100, 
                            x:float = 0.0, y:float = 0.0, theta:float = 0.0):
    """
    Generates a reference path using the turtle model.
    :return: Reference path.
    """
    turtle.reset_state(x, y, theta)
    turtle.generate_input(horizon)
    return turtle.predict(horizon)

def generate_population(horizon:int = 100, num_paths:int = 100, 
                        x:float = 0.0, y:float = 0.0, theta:float = 0.0):
    """
    Generates a population of paths using the turtle model.
    :param num_paths: Number of paths to generate.
    :return: List of paths.
    """
    inputs, paths = [], []
    for _ in range(num_paths):
        turtle.reset_state(x, y, theta)
        turtle.generate_input(horizon)
        input, path = turtle.predict(horizon)
        paths.append(path)
        inputs.append(input)
    return inputs, paths

def select_best_paths(ref_path, paths):
    """
    Evaluate a collection of candidate paths against a reference path.

    For each candidate, computes:
      - the total Euclidean deviation from the reference (sum of pointwise distances)
      - the net advance (Euclidean distance from its own start to end)

    Returns both the path that stays closest to the reference and 
    the path that makes the greatest net advance.

    :param ref_path: numpy array of shape (N, 2) representing the reference path.
    :param paths: iterable of numpy arrays, each of shape (N, 2), representing candidate paths.
    :return: a tuple of four elements:
        min_cost         – the smallest total deviation (float),
        closest_path     – the path corresponding to min_cost,
        best_advance     – the largest net advance (float),
        most_advanced_path – the path corresponding to best_advance.
    """
    min_cost = float('inf')
    closest_path = None

    best_advance = -float('inf')
    most_advanced_path    = None

    for path in paths:
        cost = compute_euc_cost(path, ref_path)
        advance = compute_advance(path)
        # If `path` advances more than any seen so far, record it
        if advance > best_advance:
            best_advance = advance
            most_advanced_path    = path

        if cost < min_cost:
            min_cost = cost
            closest_path = path

    return min_cost, closest_path, best_advance, most_advanced_path

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
        cost = compute_ctr_cost(path, input)
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return min_cost, best_path

def visualize(paths = [], ref_path = [], closest = [], most_advanced = [],  ctr_closest = []):
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
        
    if len(most_advanced) == 0:
        print("\nNo most advanced path to visualize.\n")
    else: plt.plot(closest[:, 0], closest[:, 1], 
                   color='yellow', label='Most advanced Path', linewidth=5)
    
    if len(ctr_closest) == 0:
        print("\nNo best control path to visualize.\n")
    else: plt.plot(ctr_closest[:, 0], ctr_closest[:, 1], 
                   color='blue', label='Best Control Path', linewidth=5)

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
    _, closest, _, most_advanced = select_best_paths(ref_path, paths)
    print("most_advanced_path: ", closest)
    _, ctr_closest = best_ctr_path(inputs, paths)

    end = time.time()
    print(f"\nTime taken to generate and visualize paths: {end - start:.4f} seconds\n")
    
    visualize(paths=paths,
             ref_path=ref_path,
             closest=closest,
             most_advanced=most_advanced,
             ctr_closest=ctr_closest)