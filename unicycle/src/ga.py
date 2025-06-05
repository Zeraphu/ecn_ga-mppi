import numpy as np
from ga_costs import compute_euc_cost, compute_smooth_cost

# Fraction of population to select as parents (top 20% of paths)
K_PERCENT       = 0.20  

# Crossover probability (85% chance to splice two parents; 15% copy through)
P_CROSSOVER     = 0.85  

# Population size (number of paths per generation)
# 150 is a good middle ground: large enough for diversity, small enough to evaluate quickly
POPULATION_SIZE = 150  

# Mutation standard deviation (in workspace units)
# If your x, y coordinates lie roughly in [0..10], a sigma of 0.5–1.0 means ~5–10% jitter.
# Here we assume a moderate‐sized map (e.g. 10×10), so we pick 0.5.
SIGMA           = 0.1  

# Mutation probability per waypoint (5% chance each intermediate point wiggles)
# This lets most waypoints stay fixed and only a few mutate each generation.
P_MUTATION      = 0.001  

# Maximum number of generations to run
# 50 allows substantial evolution without excessive runtime in most cases.
MAX_GENERATIONS = 50  

# Cost threshold: stop early if best path’s cost ≤ 0.1
# Pick a small number that matches your cost‐function’s scale (e.g. if cost is Euclidean error).
COST_THRESHOLD  = 0.1  



def crossover(parents, P, p_crossover=0.8):
    """
    Given a list of parent‐paths (each an (N+1, 2) numpy array), produce a new population of size P 
    by one‐point crossover with probability p_crossover.

    Args:
        parents      (list of np.ndarray): each array has shape (N+1, 2), with fixed endpoints at [0] and [N].
        P            (int): desired size of the new population.
        p_crossover  (float): probability of actually performing crossover on a chosen pair.

    Returns:
        new_population (list of np.ndarray): length‐P list of child arrays, each shape (N+1, 2).
    """
    new_population = []
    N = parents[0].shape[0] - 1  # number of segments; total rows = N+1

    while len(new_population) < P:
        # pick two parents at random
        A = parents[np.random.randint(len(parents))]
        B = parents[np.random.randint(len(parents))]

        if np.random.rand() < p_crossover:
            # choose cut point between 1 and N-1 (inclusive)
            cut = np.random.randint(1, N)

            # child1: rows 0..cut-1 from A, rows cut..N from B
            child1 = np.vstack((A[:cut], B[cut:]))
            # child2: rows 0..cut-1 from B, rows cut..N from A
            child2 = np.vstack((B[:cut], A[cut:]))

            # re‐enforce endpoints (not strictly needed if parents were valid)
            child1[0] = A[0]
            child1[N] = A[N]
            child2[0] = B[0]
            child2[N] = B[N]
        else:
            # no crossover: simply copy
            child1 = A.copy()
            child2 = B.copy()

        new_population.append(child1)
        if len(new_population) < P:
            new_population.append(child2)

    return new_population


import numpy as np

def mutate(population, sigma, p_mutation=0.01):
    """
    Given a list of paths (each an (N+1, 3) numpy array with columns [x,y,theta]),
    mutate each path’s intermediate waypoints (columns 0 and 1) with probability p_mutation 
    by adding Gaussian noise (std = sigma). Endpoints stay fixed.

    Args:
        population   (list of np.ndarray): each path has shape (N+1, 3), with [x,y,theta].
        sigma        (float): standard deviation of the Gaussian noise applied to x,y.
        p_mutation   (float): probability of mutating each intermediate waypoint (per row).

    Returns:
        new_population (list of np.ndarray): mutated population, same shapes as input.
    """
    new_population = []

    for path in population:
        N = path.shape[0] - 1              # number of segments
        new_path = path.copy()             # do not modify original
        mask = np.random.rand(N - 1) < p_mutation

        if mask.any():
            # Generate noise only for x,y columns (shape = (N-1, 2))
            noise = np.random.randn(N - 1, 2) * sigma

            # Add to x‐column for rows 1..N-1 where mask=True
            new_path[1:N, 0][mask] += noise[mask, 0]
            # Add to y‐column for rows 1..N-1 where mask=True
            new_path[1:N, 1][mask] += noise[mask, 1]

        # Re‐enforce endpoints exactly (row 0 and row N)
        new_path[0] = path[0]
        new_path[N] = path[N]

        new_population.append(new_path)

    return new_population



def select_top_K_percent(population, ref_path, k_percent):
    """
    Select the top K% of paths from `population` according to their cost.
    
    Args:
        population (list of np.ndarray): 
            List of paths, each of shape (N+1, 2).
        k_percent (float): 
            Fraction between 0 and 1 indicating the top percentage to select.
        cost_fn (callable): 
            Function that accepts a path (np.ndarray) and returns its scalar cost.
    
    Returns:
        selected_parents (list of np.ndarray): 
            The top k_percent fraction of `population`, sorted by lowest cost.
    """
    P = len(population)
    if P == 0:
        return []
    
    # 1. Compute cost for each path
    costs = np.array([compute_smooth_cost(path, ref_path) for path in population])
    
    # 2. Determine how many to select (at least 1)
    K = max(1, int(np.floor(k_percent * P)))
    
    # 3. Sort indices by ascending cost, then take first K
    sorted_indices = np.argsort(costs)
    top_indices = sorted_indices[:K]
    
    # 4. Gather those top K paths
    selected_parents = [population[i] for i in top_indices]
    return selected_parents

def run_ga(initial_paths, ref_path):
    """
    Run a GA that evolves a population of paths toward ref_path.
    
    Args:
        initial_paths (list of np.ndarray): List of shape-(T,2+) arrays—the starting population.
        ref_path       (np.ndarray):        The reference path to match, shape (T,2+).
    
    Returns:
        best_path (np.ndarray): The best path found.
        best_cost (float):      Its final cost.
    """
    # 1) Initialize
    population = initial_paths.copy()
    best_cost = float("inf")
    no_improve_count = 0
    generation = 0
    best_path = None

    while True:
        generation += 1

        # 2) Select parents: top‐K_PERCENT by cost
        cost_fn = lambda p: compute_smooth_cost(p, ref_path)
        parents = select_top_K_percent(population, ref_path, K_PERCENT)

        # 3) Crossover: produce exactly POPULATION_SIZE children
        children = crossover(parents, POPULATION_SIZE, P_CROSSOVER)

        # 4) Mutate the children
        population = mutate(
            children,
            SIGMA,
            P_MUTATION,
        )

        # 5) Evaluate new population’s costs
        costs = np.array([compute_smooth_cost(p, ref_path) for p in population])
        idx_best = int(np.argmin(costs))
        current_best = costs[idx_best]
        candidate = population[idx_best]

        # 6) Update best‐ever
        if current_best < best_cost:
            best_cost = current_best
            best_path = candidate.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 7) Check stopping criteria
        if best_cost <= COST_THRESHOLD:
            print(f"Generation {generation}: Desired accuracy reached (cost={best_cost:.4f}). Stopping.")
            break

        if generation >= MAX_GENERATIONS:
            print(f"Generation {generation}: Max generations reached (best cost={best_cost:.4f}). Stopping.")
            break

        if no_improve_count >= 10:
            print(f"Generation {generation}: No improvement in 10 generations (best cost={best_cost:.4f}). Stopping.")
            break

    return best_path, best_cost