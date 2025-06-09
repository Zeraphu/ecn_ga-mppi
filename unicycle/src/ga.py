import numpy as np

def evaluate_individual(U, x0, x_ref, alpha, dynamics_func):
    """
    Evaluate the cost J(U, X) for a single control sequence U.

    Args:
        U (np.ndarray): Array of shape (N, 2) containing control commands [(u, ω)_0, ..., (u, ω)_{N-1}].
        x0 (np.ndarray): Initial state vector.
        x_ref (np.ndarray): Reference state/position vector.
        alpha (float): Weight for control-effort penalty.
        dynamics_func (callable): Function dynamics_func(x, u) -> x_next.

    Returns:
        cost (float): Total cost J for the sequence.
        X (np.ndarray): Array of states of shape (N+1, state_dim).
    """
    N = U.shape[0]
    x_dim = x0.shape[0]
    X = np.zeros((N + 1, x_dim))
    X[0] = x0.copy()
    
    # Forward simulate states
    for k in range(N):
        X[k + 1] = dynamics_func(X[k], U[k])
    
    # State-error cost: sum of squared distances to x_ref over steps 1..N
    state_errors = X[1:] - x_ref
    J_state = np.sum(np.einsum('ij,ij->i', state_errors, state_errors))
    
    # Control-effort cost: alpha * sum of squared norms of U_k
    J_control = alpha * np.sum(np.einsum('ij,ij->i', U, U))
    
    cost = J_state + J_control
    return cost, X

def evaluate_population(U_pop, x0, x_ref, alpha, dynamics_func):
    """
    Evaluate cost for an entire population of control sequences.

    Args:
        U_pop (np.ndarray): Array of shape (P, N, 2) with P sequences of length N.
        x0 (np.ndarray): Initial state.
        x_ref (np.ndarray): Reference state/position.
        alpha (float): Control-effort weight.
        dynamics_func (callable): Function to propagate state.

    Returns:
        costs (np.ndarray): Array of shape (P,) with cost for each individual.
        trajectories (list of np.ndarray): List of P trajectories, each of shape (N+1, state_dim).
    """
    P = U_pop.shape[0]
    costs = np.zeros(P)
    trajectories = []
    
    for i in range(P):
        costs[i], X_i = evaluate_individual(U_pop[i], x0, x_ref, alpha, dynamics_func)
        trajectories.append(X_i)
    
    return costs, trajectories

def select_elites(U_pop, costs, elite_fraction=0.05):
    """
    Select the top-performing elite individuals to carry over unchanged.
    
    Args:
        U_pop (np.ndarray): Population of shape (P, N, 2).
        costs (np.ndarray): Costs array of shape (P,).
        elite_fraction (float): Fraction of population to keep as elites.
        
    Returns:
        elites (np.ndarray): Elite control sequences.
        elite_indices (np.ndarray): Indices of the elites.
    """
    P = U_pop.shape[0]
    num_elites = max(1, int(np.floor(elite_fraction * P)))
    sorted_indices = np.argsort(costs)
    elite_indices = sorted_indices[:num_elites]
    elites = U_pop[elite_indices]
    return elites, elite_indices

def tournament_selection(U_pop, costs, num_parents, tournament_size=3):
    """
    Select parents via tournament selection.
    
    Args:
        U_pop (np.ndarray): Population of shape (P, N, 2).
        costs (np.ndarray): Costs array of shape (P,).
        num_parents (int): Number of parents to select.
        tournament_size (int): Number of individuals in each tournament.
        
    Returns:
        parents (np.ndarray): Selected parent control sequences.
        parent_indices (np.ndarray): Indices of the selected parents.
    """
    P = U_pop.shape[0]
    parent_indices = []
    
    for _ in range(num_parents):
        # Randomly choose tournament competitors
        competitors = np.random.choice(P, size=tournament_size, replace=False)
        # Pick the competitor with lowest cost (best)
        best = competitors[np.argmin(costs[competitors])]
        parent_indices.append(best)
    
    parents = U_pop[parent_indices]
    return parents, np.array(parent_indices)

def roulette_wheel_selection(U_pop, costs, num_parents):
    """
    Select parents via fitness-proportionate (roulette-wheel) selection.
    
    Args:
        U_pop (np.ndarray): Population of shape (P, N, 2).
        costs (np.ndarray): Costs array of shape (P,).
        num_parents (int): Number of parents to select.
        
    Returns:
        parents (np.ndarray): Selected parent control sequences.
        parent_indices (np.ndarray): Indices of the selected parents.
    """
    # Convert cost to fitness: higher fitness = better
    # Add a small constant to avoid division by zero
    fitness = 1.0 / (costs + 1e-6)
    probabilities = fitness / np.sum(fitness)
    parent_indices = np.random.choice(len(U_pop), size=num_parents, p=probabilities, replace=True)
    parents = U_pop[parent_indices]
    return parents, parent_indices

def crossover(parent1, parent2):
    """
    Single-point row-wise crossover between two parent control sequences.
    
    Args:
        parent1, parent2 (np.ndarray): Arrays of shape (N, 2)
        
    Returns:
        child1, child2 (np.ndarray): Two offspring sequences.
    """
    N = parent1.shape[0]
    # Choose a crossover point between 1 and N-1
    cp = np.random.randint(1, N)
    # Create children by swapping rows after the crossover point
    child1 = np.vstack((parent1[:cp], parent2[cp:]))
    child2 = np.vstack((parent2[:cp], parent1[cp:]))
    return child1, child2

def mutate(child, mutation_rate=0.1, sigma=(0.1, 0.05), u_bounds=(-1,1), w_bounds=(-1,1)):
    """
    Mutate a single control sequence by adding Gaussian noise to entries.
    
    Args:
        child (np.ndarray): Array of shape (N, 2).
        mutation_rate (float): Probability of mutating each gene.
        sigma (tuple): Standard deviations for (u, omega) noise.
        u_bounds (tuple): (u_min, u_max) actuator limits for u.
        w_bounds (tuple): (w_min, w_max) actuator limits for omega.
        
    Returns:
        mutated (np.ndarray): The mutated control sequence, clipped to bounds.
    """
    N, _ = child.shape
    mutated = child.copy()
    
    # Generate a mask of which entries to mutate
    mask = np.random.rand(N, 2) < mutation_rate
    
    # Gaussian noise for u and omega
    noise = np.zeros_like(mutated)
    noise[:, 0] = np.random.normal(0, sigma[0], size=N)  # for u
    noise[:, 1] = np.random.normal(0, sigma[1], size=N)  # for omega
    
    # Apply noise where mask is True
    mutated += mask * noise
    
    # Clip to actuator limits
    mutated[:, 0] = np.clip(mutated[:, 0], u_bounds[0], u_bounds[1])
    mutated[:, 1] = np.clip(mutated[:, 1], w_bounds[0], w_bounds[1])
    
    return mutated

def simple_dynamics(x, u, dt=0.1):
    # x = [x_pos, y_pos, theta]
    # u = [v, omega]
    x_pos, y_pos, theta = x
    v, omega = u
    x_next = np.array([
        x_pos + v * np.cos(theta) * dt,
        y_pos + v * np.sin(theta) * dt,
        theta + omega * dt
    ])
    return x_next


def run_ga(
    U_pop,         
    x0,
    x_ref,
    dynamics_func=simple_dynamics,
    alpha=0.1,
    generations=50,
    elite_fraction=0.05,
    tournament_size=3,
    mutation_rate=0.1,
    sigma=(0.1, 0.05),
    u_bounds=(-1, 1),
    w_bounds=(-1, 1),
    use_roulette=False
):
    """
    Run GA starting from a user‐provided population U_pop.
    """
    P, N, _ = U_pop.shape
    history = []
    best_U = None
    best_X = None

    for gen in range(generations):
        # 1) Evaluate
        costs, trajectories = evaluate_population(
            U_pop, x0, x_ref, alpha, dynamics_func
        )
        # Track best
        best_idx = np.argmin(costs)
        history.append(costs[best_idx])
        best_U = U_pop[best_idx].copy()
        best_X = trajectories[best_idx].copy()

        # 2) Elites
        elites, _ = select_elites(U_pop, costs, elite_fraction)

        # 3) Parents
        num_offspring = P - elites.shape[0]
        if use_roulette:
            parents, _ = roulette_wheel_selection(U_pop, costs, num_offspring)
        else:
            parents, _ = tournament_selection(
                U_pop, costs,
                num_parents=num_offspring,
                tournament_size=tournament_size
            )

        # 4) Crossover & Mutate
        children = []
        for i in range(0, num_offspring, 2):
            p1 = parents[i % num_offspring]
            p2 = parents[(i+1) % num_offspring]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, mutation_rate, sigma, u_bounds, w_bounds)
            c2 = mutate(c2, mutation_rate, sigma, u_bounds, w_bounds)
            children.append(c1)
            if len(children) < num_offspring:
                children.append(c2)

        children = np.stack(children)[:num_offspring]

        # 5) New population
        U_pop = np.vstack([elites, children])

        print(f"Gen {gen+1}/{generations}, best cost = {history[-1]:.3f}")

    return best_U, best_X, history
