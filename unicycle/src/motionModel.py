#!/usr/bin/env python3

import numpy as np

class MotionModel:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.state = np.array([self.x, self.y, self.theta])

        self.v = 0.1
        self.w = 0.0
        self.input_history = np.zeros((0, 2))

    def predict(self,):
        """
        Uses a generated input from a Gaussian for v,w values,
        and then updates the state of the robot.
        """
        # Generate v and w such that the robot moves forward
        # with a small angle
        self.v = np.random.normal(0.1, 0.1)
        self.w = np.random.normal(0, 0.05)

        # Add to the input history
        if len(self.input_history) == 0:
            self.input_history = np.array([[self.v, self.w]])
        else:
            self.input_history = np.vstack((self.input_history, [self.v, self.w]))

        # Update the state using the motion model
        arg = self.theta + self.w * self.dt

        self.new_x = self.x + self.v * self.dt * np.cos(arg)
        self.new_y = self.y + self.v * self.dt * np.sin(arg)

        self.new_theta = self.theta + self.w * self.dt
        # Ensure the new theta is within the range 
        if self.new_theta > np.pi / 2:
            self.new_theta = np.pi / 2
        elif self.new_theta < -np.pi / 2:
            self.new_theta = -np.pi / 2
        # Update the state
        self.state = np.array([self.new_x, self.new_y, self.new_theta])

        # Update the current state
        self.x = self.new_x
        self.y = self.new_y
        self.theta = self.new_theta
        return self.state
    
    def gen_path(self, steps: int = 100):
        """
        Generates a path starting from [0, 0] till steps reached.
        """
        path = np.zeros((steps, 3))
        for i in range(steps):
            path[i] = self.predict()
        return path
    
    def reset(self):
        """
        Resets the state of the robot to the initial state.
        """
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.state = np.array([self.x, self.y, self.theta])
        return 
