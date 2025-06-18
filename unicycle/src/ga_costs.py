import numpy as np

def compute_path_curvature_cost(path):
    """
    Computes trajectory curvature cost based on angular deviations between path segments.
    This encourages smoother, less angular paths.
    
    Parameters:
        path (np.ndarray): Array of shape (N, 2), where each row is [x, y].
                           Expected to contain float values.
    
    Returns:
        float: Curvature smoothness penalty.
    """
    if not isinstance(path, np.ndarray) or path is None or len(path) < 3:
        return 0.0

    # Vectorized computation of segment angles
    deltas = np.diff(path, axis=0)  # shape (N-1, 2)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    angle_deltas = np.diff(angles)
    angle_deltas = np.clip(angle_deltas, -np.pi, np.pi)  # Normalize to [-π, π]
    
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
    """
    Computes the Euclidean distance cost between a path and a reference path.

    Parameters:
        path (np.ndarray): Robot path (N, 2).
        ref_path (np.ndarray): Reference path (N, 2).
    Returns:
        float: Total Euclidean distance cost.
    """
    return np.linalg.norm(path - ref_path, axis=1).sum()

def compute_ctr_cost(path, control_seq):
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
    return compute_path_curvature_cost(path) + \
           compute_control_variation_cost(control_seq) + \
           compute_control_effort_cost(control_seq)

def compute_advance(path):
    """
    Compute the Euclidean ‘advance’ of a path, defined as the
    straight‐line distance between its first and last points.
    
    :param path: array of shape (N, 2), waypoints [x, y]
    :return: Euclidean distance ||last - first||
    """
    start = path[0]    # [x_0, y_0]
    end   = path[-1]   # [x_{N-1}, y_{N-1}]
    return np.linalg.norm(end - start)

def scale_costs(costs, scale_range=(0, 1)):
    """
    Scales costs to a specified range.
    
    :param costs: List of costs to be scaled.
    :param scale_range: Tuple (min, max) defining the scaling range.
    :return: List of scaled costs.
    """
    min_val, max_val = min(costs), max(costs)
    if max_val > min_val:  # Avoid division by zero
        return [(val - min_val) / (max_val - min_val) * (scale_range[1] - scale_range[0]) + scale_range[0] for val in costs]
    else:
        # If all values are the same, return the midpoint of the scale range
        return [0.5 * (scale_range[0] + scale_range[1])] * len(costs)