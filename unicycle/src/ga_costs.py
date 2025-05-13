import numpy as np

def compute_path_curvature_cost(path):
    """
    Computes trajectory curvature cost based on angular deviations between path segments.
    This encourages smoother, less angular paths.
    
    Parameters:
        path (np.ndarray): Array of shape (N, 2), where each row is [x, y].
    
    Returns:
        float: Curvature smoothness penalty.
    """
    if len(path) < 3:
        return 0.0

    angles = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        angles.append(np.arctan2(dy, dx))

    angle_deltas = np.diff(angles)
    angle_deltas = (angle_deltas + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
    
    return np.linalg.norm(angle_deltas)


def compute_control_variation_cost(control_seq):
    """
    Computes control effort smoothness cost based on changes in linear and angular velocity.
    
    Parameters:
        control_seq (np.ndarray): Array of shape (N, 2), each row is [v, w].
    
    Returns:
        float: Smoothness penalty from velocity changes.
    """
    if len(control_seq) < 2:
        return 0.0

    deltas = np.diff(control_seq, axis=0)
    return np.sum(np.square(deltas))


def compute_control_effort_cost(control_seq):
    """
    Computes cumulative control effort (energy cost) from linear and angular velocities.
    
    Parameters:
        control_seq (np.ndarray): Array of shape (N, 2), each row is [v, w].
    
    Returns:
        float: Total control effort cost.
    """
    return np.sum(np.square(control_seq[:, 0]) + np.square(control_seq[:, 1]))

def compute_euc_cost(path, ref_path):
    return np.linalg.norm(path - ref_path, axis=1).sum()

def compute_advance(path) -> float:
    """
    Compute the Euclidean ‘advance’ of a path, defined as the
    straight‐line distance between its first and last points.
    
    :param path: array of shape (N, 2), waypoints [x, y]
    :return: Euclidean distance ||last - first||
    """
    start = path[0]    # [x_0, y_0]
    end   = path[-1]   # [x_{N-1}, y_{N-1}]
    return np.linalg.norm(end - start)

def compute_ctr_cost(path, control_seq, 
                       alpha=1.0, 
                       beta=1.0, 
                       gamma=1.0,):
    """
    Combines multiple cost terms: curvature, control smoothness, and energy.
    
    Parameters:
        path (np.ndarray): Robot path (N, 2)
        control_seq (np.ndarray): Control inputs (N, 2)
        alpha (float): Weight for curvature cost.
        beta  (float): Weight for control variation cost.
        gamma (float): Weight for energy effort cost.
    
    Returns:
        float: Combined total cost.
    """
    curvature = compute_path_curvature_cost(path)
    variation = compute_control_variation_cost(control_seq)
    energy = compute_control_effort_cost(control_seq)

    return alpha * curvature + beta * variation + gamma * energy