#!/usr/bin/env python3
"""
Genetic algorithm for selecting the best path with all cost functions (closest path and best control path).
"""

from motionModel import turtleModel
import pop_generator
import ga_costs
import numpy as np
import matplotlib.pyplot as plt
import time

class MPPI_GA():
    def __init__(self, horizon:int = 100, num_paths:int = 100,):
        """
        Initializes the MPPI_GA class with the given parameters.
        :param horizon: Number of steps in the prediction horizon.
        :param num_paths: Number of paths to generate.
        """
        self.horizon = horizon
        self.num_paths = num_paths
    
    def compute_costs(self, ref_path, paths, inputs):
        """
        Computes the costs for the given paths and reference path.
        Scales the costs to the same range before summing them.
        :param ref_path: Reference path.
        :param paths: List of paths.
        :param inputs: List of inputs.
        :return: Scaled total costs, scaled control costs, and scaled Euclidean costs.
        """
        # Compute individual costs
        ctr_costs = [ga_costs.compute_ctr_cost(path, inp) for path, inp in zip(paths, inputs)]
        euc_costs = [ga_costs.compute_euc_cost(path, ref_path) for path in paths]

        # Scale costs to the range [0, 1]
        def scale_to_range(values):
            min_val, max_val = min(values), max(values)
            if max_val > min_val:  # Avoid division by zero
                return [(val - min_val) / (max_val - min_val) for val in values]
            else:
                # If all values are the same, return 0.5 for all
                return [0.5] * len(values)

        scaled_ctr_costs = scale_to_range(ctr_costs)
        scaled_euc_costs = scale_to_range(euc_costs)

        # Combine scaled costs
        total_costs = [ctr + euc for ctr, euc in zip(scaled_ctr_costs, scaled_euc_costs)]

        return total_costs, scaled_ctr_costs, scaled_euc_costs
    
if __name__ == "__main__":
    turtle = turtleModel()
    
    _, ref_path = pop_generator.generate_reference_path(horizon=100)
    inputs, paths = pop_generator.generate_population(horizon=100, num_paths=100)
    
    # Initialize the MPPI_GA class
    mppi_ga = MPPI_GA(horizon=100, num_paths=100)
    costs, ctr_costs, euc_costs = mppi_ga.compute_costs(ref_path, paths, inputs)
    
    # Find the best path based on different costs
    best_path_index = np.argmin(costs)
    best_path = paths[best_path_index]

    best_euc_index = np.argmin(euc_costs)
    best_euc_path = paths[best_euc_index]

    best_ctr_index = np.argmin(ctr_costs)
    best_ctr_path = paths[best_ctr_index]
    
    plt.figure(figsize=(10, 6))
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], color='gray', alpha=0.5)
    plt.plot(ref_path[:, 0], ref_path[:, 1], label='Reference Path', color='blue', linestyle='--')
    plt.plot(best_path[:, 0], best_path[:, 1], label='Best Path', color='red', linewidth=5)
    plt.plot(best_euc_path[:, 0], best_euc_path[:, 1], label='Best Euclidean Path', color='green')
    plt.plot(best_ctr_path[:, 0], best_ctr_path[:, 1], label='Best Control Path', color='orange')
    plt.legend()
    plt.title('Reference Path vs Best Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid()
    plt.show()