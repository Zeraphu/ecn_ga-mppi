#!/usr/bin/env python3
"""
Genetic algorithm for selecting the best path with all cost functions (closest path and best control path).
"""

import motionModel
import utils
import ga_costs
import numpy as np
import matplotlib.pyplot as plt
import time

class MPPI_GA():
    def __init__(self, 
                 horizon:int = 100, 
                 pop_size:int = 100,
                 motion_model = motionModel.turtleModel()
                 ):
        """
        Initializes the MPPI_GA class with the given parameters.
        :param horizon: Number of steps in the prediction horizon.
        :param num_paths: Number of paths to generate.
        """
        if horizon <= 2:
            raise ValueError("horizon must be greater than 2.")
        self.horizon = horizon
        self.num_paths = pop_size
        self.dynamics = motion_model

        self.population = np.empty((0, self.horizon, 2))  # Initialize population
        self.current_state = np.array([0.0, 0.0, 0.0])  # Initial state [x, y, theta]
        self.costs = np.empty((0,))  # Initialize costs
        self.elites_fraction = 0.05 
    
    def compute_costs(self, ref_path, paths, inputs):
        """
        Computes the costs (i.e., euc, ctr, advance) for the given generation, scales them using ga_costs.scale_costs,
        sums them for each path, and returns the total costs.
        """
        euc_costs = np.array([ga_costs.compute_euc_cost(path, ref_path) for path in paths])
        ctr_costs = np.array([ga_costs.compute_ctr_cost(path, inputs[i]) for i, path in enumerate(paths)])
        advance_costs = np.array([ga_costs.compute_advance(path) for path in paths])
        
        # Scale costs
        euc_costs = ga_costs.scale_costs(euc_costs)
        ctr_costs = ga_costs.scale_costs(ctr_costs)
        advance_costs = ga_costs.scale_costs(advance_costs)

        if len(euc_costs) != len(ctr_costs) or len(euc_costs) != len(advance_costs):
            raise ValueError("Cost arrays must have the same length.")
        # Sum costs for each path
        total_costs = np.sum([np.dot(0.5, euc_costs), np.dot(0.8, ctr_costs), np.dot(0.8, advance_costs)], axis=0)
        return np.array(total_costs)
    
    def elite_selection(self, inputs, costs, elite_fraction=0.05):
        """
        Selects the top-performing elite individuals to carry over unchanged.
        
        Args:
            inputs (np.ndarray): Population of shape (P, N, 2).
            costs (np.ndarray): Costs array of shape (P,).
            elite_fraction (float): Fraction of population to keep as elites.
            
        Returns:
            elites (np.ndarray): Elite control sequences.
            elite_indices (np.ndarray): Indices of the elites.
        """
        P = inputs.shape[0]
        num_elites = max(1, int(np.floor(elite_fraction * P)))
        sorted_indices = np.argsort(costs)
        elite_indices = sorted_indices[:num_elites]
        elites = inputs[elite_indices]
        return elites, elite_indices

    def crossover_mutate(self, parents, mutation_rate=0.1):
        """
        Performs uniform crossover and mutation to generate new children from parents.
        
        Args:
            parents (np.ndarray): Parent population of shape (P, N, 2).
            mutation_rate (float): Probability of applying mutation to a child.
            
        Returns:
            children (np.ndarray): New population of shape (P, N, 2).
        """
        num_parents = parents.shape[0]
        children = np.empty((num_parents, self.horizon, 2))
        
        for i in range(num_parents):
            parent1_idx = np.random.randint(num_parents)
            parent2_idx = np.random.randint(num_parents)
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            
            # Uniform crossover
            mask = np.random.rand(self.horizon, 2) < 0.5
            children[i] = np.where(mask, parent1, parent2)
            
            # Mutation
            if np.random.rand() < mutation_rate:
                mutation = np.random.normal(0, 0.1, (self.horizon, 2))
                children[i] += mutation
                
        return children

    def run(self, 
            current_state,
            ref_path):
        """
        Runs the genetic algorithm to find the best path and control sequence.
        """
        generation = 0
        cost_history = []
        best_U = None
        best_X = None
        
        # 1. Regenerate population (Assumes input is set in the motionModel)
        self.population, paths = utils.generate_population(self.horizon, self.num_paths, 
                                    self.current_state[0], 
                                    self.current_state[1], 
                                    self.current_state[2],
                                    self.dynamics)
        
        start = time.time()
        while True:
            generation += 1
            # 2. Compute costs
            self.costs = self.compute_costs(ref_path, paths, self.population)
            cost_history.append(np.mean(self.costs))
            # print(f"Generation {generation}, Min Cost: {np.min(self.costs):.4f}")

            # 3. Check for convergence
            if generation > 1000 or np.min(self.costs) < 0.15 or time.time() - start > 0.5:
                best_U = np.clip(self.population[np.argmin(self.costs)], self.dynamics.min_v, self.dynamics.max_v)
                best_X = paths[np.argmin(self.costs)]
                return best_U, best_X, cost_history, paths

            # 3. Select elites
            elites, elite_indices = self.elite_selection(self.population, self.costs, self.elites_fraction)

            # 4. Crossover and mutation to create new population
            children = self.crossover_mutate(np.delete(self.population, elite_indices, axis=0))

            # 5. Combine elites and children to form new population
            self.population = np.vstack((elites, children))
            self.population, paths = utils.generate_population_from_inputs(
                self.horizon, self.num_paths,
                self.current_state[0], 
                self.current_state[1], 
                self.current_state[2],
                self.population, 
                self.dynamics)

if __name__ == "__main__":
    # Example usage
    horizon = 10
    pop_size = 10
    start = time.time()
    turtle = motionModel.turtleModel(dt=0.1,
                                    min_v=-0.1,
                                    max_v=0.2,
                                    max_w=0.4,
                                    v_std=0.02,
                                    w_std=0.2/3)
    mppi_ga = MPPI_GA(horizon, pop_size, turtle)
    
    mppi_ga.dynamics.generate_input(horizon)
    current_state = np.array([0.0, 0.0, 0.0])  # Initial state [x, y, theta]
    _, ref_path = utils.generate_path(horizon,
                                      current_state[0], 
                                      current_state[1], 
                                      current_state[2], 
                                      mppi_ga.dynamics)
    
    best_U, best_X, cost_history, paths = mppi_ga.run(current_state, ref_path)
    print(f"Best control sequence: {best_U}")

    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Generations: {len(cost_history)}")
    # print("Best control sequence shape:", best_U.shape)
    # print(paths.shape)
    # print(ref_path.shape, "reference path shape")
    # print(best_X.shape, "best path shape")

    utils.visualize(paths = paths, ref_path = ref_path, ga_path= best_X)