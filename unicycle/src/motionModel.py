#!/usr/bin/env python3

import numpy as np

class turtleModel():
    def __init__(self, 
                 dt: float = 0.1,
                 min_v: float = 0.1,
                 max_v: float = 0.5, 
                 max_w: float = 1.4572,
                 v_std: float = 0.2,
                 w_std: float = 0.4,):
        """
        Initializes the turtle model with the given parameters.
        :param dt: Time step for the simulation.
        :param min_v: Minimum velocity.
        :param max_v: Maximum velocity.
        :param max_w: Maximum angular velocity.
        :param v_std: Standard deviation for velocity.
        :param w_std: Standard deviation for angular velocity.
        """
        self.dt = dt
        self.min_v = min_v
        self.max_v = max_v
        self.v_std = v_std
        self.w_std = w_std
        self.max_w = max_w
        self.U = np.empty((0, 2))   
        self.X = np.empty((0, 3))

    def reset_state(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """
        Sets the state of the turtle model.
        :param x: x position, default=0.0.
        :param y: y position, default=0.0.
        :param theta: orientation, default=0.0.
        """
        self.X = np.array([[x, y, theta]])  # State vector
        return self.X
    
    def reset_state_from_pop(self, state: np.ndarray):
        """
        Sets the state of the turtle model from a numpy array.
        :param state: State vector as a numpy array of shape (3,).
        """
        if state.shape != (3,):
            raise ValueError("State must be a numpy array of shape (3,).")
        self.X = np.array([state])
        return self.X
    
    def reset_input(self, v: float = 0.0, w: float = 0.0):
        """
        Sets the input for the turtle model.
        :param v: Linear velocity, default=0.0.
        :param w: Angular velocity, default=0.0.
        """
        self.U = np.array([[v, w]])
        return self.U
    
    def reset_input_from_pop(self, input: np.ndarray):
        """
        Sets the input for the turtle model from a numpy array.
        :param input: Input vector as a numpy array of shape (2,).
        """
        self.U = np.array(input)
        return self.U

    def generate_input(self, max_steps: int = 100):
        """
        Resets U and generates new random inputs for the turtle model.
        :param max_steps: Maximum number of steps to generate. (Should be equal to the horizon)
        :return: Input vector randomized from a Gaussian distribution.
        """
        self.U = np.empty((0, 2))  # Reset U to empty array
        v_values = np.random.normal((self.max_v + self.min_v) / 2, self.v_std, max_steps)
        v_values = np.clip(v_values, self.min_v, self.max_v)  # Clamp values to [min_v, max_v]
        w_values = np.random.normal(0, self.max_w, max_steps)
        w_values = np.clip(w_values, -self.max_w, self.max_w)  # Clamp values to [-max_w, max_w]
        self.U = np.vstack((self.U, np.column_stack((v_values, w_values))))
        return self.U
    
    def get_jacobian(self):
        """
        Computes the Jacobian matrix of the turtle model.
        :return: Jacobian matrix A and B.
        """
        phi = self.X[0, 2] + self.dt * self.U[0, 1]
        A = np.array([[1, 0, -self.dt * self.U[0, 0] * np.sin(phi)],
                      [0, 1, self.dt * self.U[0, 0] * np.cos(phi)],
                      [0, 0, 1]])
        
        B = np.array([[self.dt * np.cos(phi), -0.5 * self.dt**2 * self.U[0, 0] * np.sin(phi)],
                      [self.dt * np.sin(phi), 0.5 * self.dt**2 * self.U[0, 0] * np.cos(phi)],
                      [0, self.dt]])
        return A, B
    
    def predict(self, max_steps: int = 100):
        """
        Predicted path of max_steps length using the current U.
        :return: Predicted state vector.
        """
        A, B = self.get_jacobian()
        if len(self.U) < max_steps:
            raise ValueError(f"{[turtleModel.__name__]}:\tInsufficient inputs generated: expected {max_steps}, got {len(self.U)}")
            
        for i in range(max_steps):
            self.X = np.vstack((self.X, A @ self.X[-1] + B @ self.U[i]))
        return self.U, self.X
        
# Example usage
# The `turtleModel` class simulates the motion of a unicycle-like robot.
# This script demonstrates setting an initial state, predicting a path, 
# and visualizing the predicted trajectory.

if __name__ == "__main__":
    turtle = turtleModel(dt=0.1,
                         min_v=0.1,
                         max_v=0.5,
                         max_w=1.4572,
                         v_std=0.2,
                         w_std=0.4)

    import time
    start = time.time()
    turtle.reset_state(0, 0, 0)
    turtle.generate_input(10)
    ref_input, ref_state = turtle.predict(10)
    end = time.time()
    print(f"\nTime taken to generate path: {end - start:.4f} seconds\n")

    print("Reference Input:\n", ref_input)
    # Plot the ref_state
    import matplotlib.pyplot as plt
    plt.plot(ref_state[:, 0], ref_state[:, 1])
    plt.title("Predicted Path")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis('equal')
    plt.grid()
    plt.show()