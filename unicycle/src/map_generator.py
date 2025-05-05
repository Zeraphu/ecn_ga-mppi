#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random

def get_input():
    """
    Gets input from the user for the occupancy map generation.
    map height, width, resolution, and obstacle density.
    """
    print("Enter the following parameters for occupancy map generation:")
    height = int(input("Map Height (in pixels): "))
    width = int(input("Map Width (in pixels): "))
    resolution = float(input("Map Resolution (in meters/pixel): ")) # NOTE: This is not used in the current implementation, but will be useful with ROS.
    obstacle_density = float(input("Obstacle Density (0-1): "))

    # Validate inputs
    if height <= 0 or width <= 0:
        raise ValueError("Height and Width must be positive integers.")
    if resolution <= 0:
        raise ValueError("Resolution must be a positive number.")
    if not (0 <= obstacle_density <= 1):
        raise ValueError("Obstacle Density must be between 0 and 1.")

    return [height, width, resolution, obstacle_density]

def obstacle_generator():
    """
    Generates a random circular obstacles of random radius and position.
    :return: Random radius and position of the obstacle.
    """
    radius = random.uniform(0.1, 0.5)  # Random radius between 0.1 and 0.5 meters
    x = random.uniform(0, 1)  # Random x position between 0 and 1
    y = random.uniform(0, 1)  # Random y position between 0 and 1
    return radius, (x, y)

def generate_occupancy_map(height, width, resolution, obstacle_density):
    """
    Generates an occupancy map based on the given parameters.
    :param height: Height of the map in pixels.
    :param width: Width of the map in pixels.
    :param resolution: Resolution of the map in meters/pixel.
    :param obstacle_density: Density of obstacles in the map (0-1).
    :return: Occupancy map as a 2D numpy array.
    """
    # Initialize the occupancy map with zeros (free space)
    occupancy_map = np.zeros((height, width), dtype=np.uint8)
    num_obstacles = int(height * width * obstacle_density)
    # Randomly place obstacles in the map
    for _ in range(num_obstacles):
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        occupancy_map[x, y] = 1  # Mark as occupied
    
    # Generate a boarder around the map
    border_thickness = int(0.02 * min(height, width))  # 5% of the smaller dimension
    occupancy_map[:border_thickness, :] = 1
    occupancy_map[-border_thickness:, :] = 1
    occupancy_map[:, :border_thickness] = 1
    occupancy_map[:, -border_thickness:] = 1

    return occupancy_map

def save_occupancy_map(occupancy_map, height, width, resolution, obstacle_density):
    """
    Saves the occupancy map to a file in maps/ folder in .npy format.
    :param occupancy_map: Occupancy map to save.
    :param filename: Filename to save the map.
    """
    if input("Do you want to save the occupancy map? (y/n): ").lower() != 'y':
        print("Occupancy map not saved.")
        return None
    filename = input("Enter filename for the occupancy map (without extension): ")
    filename = f"unicycle/maps/{filename}.npy" 
    np.save(filename, occupancy_map)
    # Save the map metadata as .yaml with the same name
    yaml_filename = filename.replace('.npy', '.yaml')
    with open(yaml_filename, 'w') as f:
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"obstacle_density: {obstacle_density}\n")
    print(f"Occupancy map metadata saved to {yaml_filename}")
    print(f"Occupancy map saved to {filename}")
    return filename

def visualize_occupancy_map(occupancy_map):
    """
    Visualizes the generated occupancy map.
    :param occupancy_map: Occupancy map to visualize.
    """
    # plt.imshow(occupancy_map, cmap='gray')
    # plot the occupancy map, with obstacles in black and free space in white. Plott in inverted colors
    plt.figure(figsize=(10, 10))
    plt.imshow(occupancy_map, cmap='gray_r', origin='lower')
    plt.colorbar(label='Occupancy')
    plt.clim(0, 1)  # Set color limits to 0 and 1
    plt.title("Occupancy Map")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if input("\nDo you want to generate a new occupancy map? (y/n): ").lower() == 'y':
        print("\nGenerating a new occupancy map...\n")
        # Get input from the user
        params = get_input()
        occupancy_map = generate_occupancy_map(*params)
        map_file = save_occupancy_map(occupancy_map, *params)

        # Get the occupancy map from map_file and visualize it
        if map_file is not None:
            print(f"Loading occupancy map from {map_file}")
            # Load the occupancy map from the file
            # Note: The filename should be the same as the one used to save the map
            # and should be in the same directory as this script.
            saved_occupancy_map = np.load(map_file)
            visualize_occupancy_map(saved_occupancy_map)
        else:
            visualize_occupancy_map(occupancy_map)
    elif input("Do you want to load an existing occupancy map? (y/n): ").lower() == 'y':
        print("\nLoading an existing occupancy map...\n")
        map_file = input("Enter the filename of the occupancy map (without extension): ")
        map_file = f"unicycle/maps/{map_file}.npy"
        # Load the occupancy map from the file
        saved_occupancy_map = np.load(map_file)
        visualize_occupancy_map(saved_occupancy_map)
    else:
        print("Invalid input.Exiting...\n")
        exit(0)