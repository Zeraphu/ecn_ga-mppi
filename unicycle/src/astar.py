#!usr/bin/env python3

"""
A* Pathfinding Algorithm for occupancy grid maps saved in a .npy file. 

It reads the map metadata from a .yaml file with the same name. This script implements the A* algorithm to find the shortest path from a start point to an end point in a 2D occupancy grid map. The occupancy grid map is represented as a 2D numpy array, where 0 indicates free space and 1 indicates an obstacle.
The script includes functions for loading the map, visualizing it, and performing the A* search.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

class Astar:
    def __init__(self, map_file):
        """
        Initializes the A* algorithm with the given map file.
        :param map_file: Path to the .npy file containing the occupancy grid map.
        """
        self.map_file = map_file
        self.occupancy_map = None
        self.start = None
        self.end = None
        self.path = []
        self.load_map()
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.path_length = 0
        self.path_found = False
        self.path_length = 0
    
    def load_map(self):
        """
        Loads the occupancy map from a .npy file and its metadata from a .yaml file.
        The .yaml file should have the same name as the .npy file.
        """
        self.occupancy_map = np.load(self.map_file)
        # Load metadata from the corresponding .yaml file
        yaml_file = self.map_file.replace('.npy', '.yaml')
        with open(yaml_file, 'r') as f:
            metadata = f.read()
            # Parse metadata (height, width, resolution, obstacle density)
            self.height = int(metadata.split('height: ')[1].split('\n')[0])
            self.width = int(metadata.split('width: ')[1].split('\n')[0])
            self.resolution = float(metadata.split('resolution: ')[1].split('\n')[0])
            self.obstacle_density = float(metadata.split('obstacle_density: ')[1].split('\n')[0])
        print(f"\nMap loaded from {self.map_file}")
        print(f"Map metadata loaded from {yaml_file}")
        print(f"Map dimensions: {self.height} x {self.width}")
        print(f"Resolution: {self.resolution} m/pixel")
        print(f"Obstacle density: {self.obstacle_density}\n")

    def set_start_end(self, start, end):
        """
        Sets the start and end points for the A* search.
        :param start: Tuple (x, y) representing the start point.
        :param end: Tuple (x, y) representing the end point.
        """
        self.start = start
        self.end = end
        print(f"\nStart point set to {self.start}")
        print(f"End point set to {self.end}\n")
        if not self.is_valid_point(self.start) or not self.is_valid_point(self.end):
            raise ValueError("Start or end point is invalid (out of bounds or occupied).\n")
        if self.occupancy_map[self.start[0], self.start[1]] == 1:
            raise ValueError("Start point is occupied.\n")
        if self.occupancy_map[self.end[0], self.end[1]] == 1:
            raise ValueError("End point is occupied.\n")
    
    def is_valid_point(self, point):
        """
        Checks if a point is valid (within bounds and not occupied).
        :param point: Tuple (x, y) representing the point.
        :return: True if the point is valid, False otherwise.
        """
        x, y = point
        return 0 <= x < self.height and 0 <= y < self.width and self.occupancy_map[x, y] == 0
    
    def heuristic(self, a, b):
        """
        Heuristic function for A* algorithm (Euclidean distance).
        :param a: Tuple (x, y) representing the first point.
        :param b: Tuple (x, y) representing the second point.
        :return: Euclidean distance between the two points.
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def astar_search(self):
        """
        Performs the A* search algorithm to find the shortest path from start to end.
        :return: List of tuples representing the path from start to end.
        """
        self.start_time = time.time()
        open_set = PriorityQueue()
        open_set.put((0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.end)}
        
        while not open_set.empty():
            current = open_set.get()[1]
            if current == self.end:
                self.path_found = True
                self.reconstruct_path(came_from, current)
                break
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.end)
                    if neighbor not in [i[1] for i in open_set.queue]:
                        open_set.put((f_score[neighbor], neighbor))
        
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        if not self.path_found:
            print("No path found.\n")
            return []
        
        print(f"Path found with length {self.path_length} in {self.execution_time:.2f} seconds.\n")
        return self.path
    
    def get_neighbors(self, point):
        """
        Returns the valid neighbors of a given point in the occupancy map.
        :param point: Tuple (x, y) representing the point.
        :return: List of valid neighbors (tuples) of the point.
        """
        x, y = point
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_valid_point((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstructs the path from the end point to the start point using the came_from dictionary.
        :param came_from: Dictionary mapping each point to its predecessor.
        :param current: Current point (end point).
        :return: List of tuples representing the path from start to end.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        self.path = total_path
        self.path_length = len(total_path)
        print(f"Path length: {self.path_length}\n")
        return total_path

    def save_path(self, filename):
        """
        Saves the path as a .npy file in the maps folder.
        :param filename: Path to the .npy file to save the path.
        """
        if not self.path_found:
            print("No path found to save.\n")
            return
        np.save(filename, self.path)
        print(f"Path saved to {filename}\n")

    def visualize_path(self):
        """
        Visualizes the occupancy map and the path found by the A* algorithm.
        """
        plt.imshow(self.occupancy_map, cmap='gray_r', origin='lower')
        if self.path_found:
            path_x, path_y = zip(*self.path)
            plt.plot(path_y, path_x, color='red', linewidth=2)
            plt.scatter(self.start[1], self.start[0], color='green', label='Start', s=100)
            plt.scatter(self.end[1], self.end[0], color='blue', label='End', s=100)
            plt.legend()
        plt.title("A* Pathfinding")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

# Example usage
def main():
    """
    Main function to run the A* algorithm.
    """
    # Load the map from the occupancy grid file and its metadata from maps folder
    map_name = input("Enter the name of the occupancy map file (without extension): ")
    map_file = f"unicycle/maps/{map_name}.npy"

    # Initialize A* algorithm with the generated map
    astar = Astar(map_file)
    
    # Set start and end points from the user
    start_x = int(input("Enter start x coordinate: "))
    start_y = int(input("Enter start y coordinate: "))
    end_x = int(input("Enter end x coordinate: "))
    end_y = int(input("Enter end y coordinate: "))
    start = (start_x, start_y)
    end = (end_x, end_y)
    astar.set_start_end(start, end)
    
    # Perform A* search
    astar.astar_search()
    
    # Save the path to a .npy file with the same name as the map file with start and end coordinates
    if input("Do you want to save the path? (y/n): ").lower() == 'y':
        path_file = f"{map_name}_path_{start_x}_{start_y}_{end_x}_{end_y}"
        path_file = f"unicycle/maps/{path_file}.npy"
        astar.save_path(path_file)
    else:
        print("Path not saved.\n")
    
    # Visualize the occupancy map and the path found by A*
    astar.visualize_path()

if __name__ == "__main__":
    main()