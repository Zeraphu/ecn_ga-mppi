#!/usr/bin/env python3

import motionModel
import numpy as np
import matplotlib.pyplot as plt
import time
from ga_costs import compute_ctr_cost, compute_euc_cost
import ga

def generate_path(horizon:int = 100, 
                  x:float = 0.0, y:float = 0.0, theta:float = 0.0,
                  model = motionModel.turtleModel()):
    """
    Generates a reference path using the turtle model.
    :return: Reference path.
    """
    model.reset_state(x, y, theta)
    return model.predict(horizon)

def generate_population(horizon:int = 100, 
                        num_paths:int = 100, 
                        x:float = 0.0, y:float = 0.0, theta:float = 0.0,
                        model = motionModel.turtleModel()):
    """
    Generates a population of paths using the turtle model.
    :param num_paths: Number of paths to generate.
    :return: Tuple of numpy arrays (inputs, paths).
    """
    inputs, paths = [], []
    for _ in range(num_paths):
        model.generate_input(horizon)
        input, path = generate_path(horizon, x, y, theta, model)
        paths.append(path)
        inputs.append(input)
    return np.array(inputs), np.array(paths)

def generate_population_from_inputs(horizon,
                                    num_paths,
                                    x, y, theta,
                                    inputs, 
                                    model = motionModel.turtleModel()):
    """
    Generates a population of paths from given inputs using the turtle model.
    :param horizon: Number of steps in the prediction horizon.
    :param num_paths: Number of paths to generate.
    :param x: Initial x position.
    :param y: Initial y position.
    :param theta: Initial orientation.
    :param inputs: List of control inputs.
    :param model: Motion model to use for path generation.
    :return: Tuple of numpy arrays (inputs, paths).
    """
    if len(inputs) != num_paths:
        raise ValueError("Number of inputs must match num_paths.")
    paths = []
    for input in inputs:
        model.reset_input_from_pop(input)
        input, path = generate_path(horizon, x, y, theta, model)
        paths.append(path)
    return np.array(inputs), np.array(paths)

def closest_path(ref_path, paths):
    """
    Finds the closest path to the reference path based on the cost function.
    NOTE: param paths has to be a list of paths!
    :param ref_path: Reference path.
    :param paths: List of paths.
    :return: minimum cost and closest path.
    """
    min_cost = float('inf')
    closest_path = None

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
    min_cost = float('inf')
    best_path = None
    for input, path in zip(inputs, paths):
        cost = compute_ctr_cost(path, input)
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return min_cost, best_path

def most_advance_path(paths):
    """
    Finds the path with the most advance.
    :param paths: List of paths.
    :return: Path with the most advance.
    """
    max_advance = -1
    best_path = None
    for path in paths:
        advance = np.linalg.norm(path[-1] - path[0])
        if advance > max_advance:
            max_advance = advance
            best_path = path
    return max_advance, best_path

def visualize(paths = [], ref_path = [], closest = [], ctr_closest = [], adv_path = [], ga_path = []):
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
    for path in paths:
        plt.plot(path[:, 0], path[:, 1])

    if len(ref_path) == 0:
        raise ValueError("\nNo reference path to visualize.\n")
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
        
    if len(adv_path) == 0:
        print("\nNo most advance path to visualize.\n")
    else: plt.plot(adv_path[:, 0], adv_path[:, 1], 
                   color='green', label='Most Advance Path', linewidth=5)
    
    if len(ga_path) == 0:
        print("\nNo GA path to visualize.\n")
    else: plt.plot(ga_path[:, 0], ga_path[:, 1], 
                   color='purple', label='Best GA Path', linewidth=5)

    plt.legend()
    plt.title("Generated Paths")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    start = time.time()

    turtle = motionModel.turtleModel()
    horizon, num_paths = 100, 100
    turtle.generate_input(horizon)

    _, ref_path = generate_path(horizon=horizon, model =turtle)
    inputs, paths = generate_population(horizon=horizon, num_paths=num_paths, model=turtle)
    _, closest = closest_path(ref_path, paths)
    _, ctr_closest = best_ctr_path(inputs, paths)
    _, most_adv_path = most_advance_path(paths)

    population = np.array(inputs)
    x0=paths[0][0]
    x_ref=ref_path[-1]
    best_U, best_X, _ = ga.run_ga(population, x0, x_ref)
    print(best_U.shape, np.array(paths).shape) # type: ignore

    end = time.time()
    print(f"\nTime taken to generate and visualize paths: {end - start:.4f} seconds\n")
    
    visualize(paths=paths,
             ref_path=ref_path,
             closest=closest,
             ctr_closest=ctr_closest,
             adv_path=most_adv_path,)