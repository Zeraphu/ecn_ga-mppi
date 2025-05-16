#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from pop_generator import generate_population

# Load occupancy map and compute distance transform
def load_distance_map(map_path):
    occupancy_map = np.load(map_path)  # Binary map: 0=free, 1=obstacle
    obstacle_map = occupancy_map == 1
    dist_map = distance_transform_edt(~obstacle_map)  # Distance from each free cell to the nearest obstacle
    return dist_map

# Convert world coordinates to map indices
def world_to_map(x, y, map_info):
    resolution = map_info["resolution"]
    origin = map_info["origin"]  # [x, y]
    map_x = int((x - origin[0]) / resolution)
    map_y = int((y - origin[1]) / resolution)
    return map_x, map_y

# Compute average distance from nearest obstacle for each path
def compute_avg_distances(paths, dist_map, map_info):
    avg_distances = []
    for path in paths:
        total_dist = 0
        valid_points = 0
        for x, y in path:
            map_x, map_y = world_to_map(x, y, map_info)
            if 0 <= map_x < dist_map.shape[1] and 0 <= map_y < dist_map.shape[0]:
                total_dist += dist_map[map_y, map_x]
                valid_points += 1
        avg_distances.append(total_dist / valid_points if valid_points else float('inf'))
    return avg_distances

if __name__ == "__main__":
    # === CONFIG ===
    map_npy = "maps/demo_map1.npy"            # Path to occupancy map
    map_yaml = "maps/demo_map1.yaml"          # Path to metadata
    start_x, start_y, theta = 0.0, 0.0, 0.0
    horizon, num_paths = 100, 100

    # === Load map ===
    import yaml
    with open(map_yaml, 'r') as f:
        map_info = yaml.safe_load(f)
        map_info = {
            "resolution": map_info["resolution"],
            "origin": map_info["origin"][:2]  # Assume 2D origin [x, y]
        }

    dist_map = load_distance_map(map_npy)

    # === Generate population ===
    _, paths = generate_population(horizon=horizon, num_paths=num_paths, 
                                   x=start_x, y=start_y, theta=theta)

    # === Evaluate ===
    avg_distances = compute_avg_distances(paths, dist_map, map_info)

    # === Print Results ===
    for i, dist in enumerate(avg_distances):
        print(f"Path {i}: Avg Distance from Obstacles = {dist:.3f} meters")

    # Optional: Visualize top N safest paths
    top_n = 5
    best_indices = np.argsort(avg_distances)[-top_n:]  # Top N with highest avg distance from obstacles
    for idx in best_indices:
        plt.plot(paths[idx][:, 0], paths[idx][:, 1], label=f"Path {idx} ({avg_distances[idx]:.2f}m)")
    plt.title("Top N Safest Paths (by Avg Distance from Obstacles)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()
