#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class underwaterRobotModel:
    def __init__(self, 
                 dt: float = 0.1,
                 mass: float = 30.0,
                 inertia_z: float = 6.0,
                 max_thrust: float = 100.0,
                 hydro_params: dict = None):
        """
        Underwater robot model with 4 thrusters
        :param dt: Time step (seconds)
        :param mass: Total mass including added mass (kg)
        :param inertia_z: Moment of inertia about z-axis (kg·m²)
        :param max_thrust: Maximum thrust per thruster (N)
        :param hydro_params: Hydrodynamic parameters dictionary
        """
        self.dt = dt
        self.mass = mass
        self.inertia_z = inertia_z
        self.max_thrust = max_thrust
        
        # Hydrodynamic parameters (simplified)
        self.hydro = hydro_params or {
            'X_udot': 5.0, 'Y_vdot': 5.0, 'Z_wdot': 5.0,
            'X_u': 10.0, 'Y_v': 10.0, 'Z_w': 10.0, 'N_r': 4.0
        }
        
        # Thruster allocation matrix (from earlier derivation)
        self.B = np.array([
            [np.sqrt(2)/2, np.sqrt(2)/2, 0, 0],
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0],
            [0, 0, 1, -1],
            [0.5, -0.5, 0, 0]  # Simplified moment arm
        ])
        
        # State: [x, y, z, ψ, u, v, w, r]
        self.X = np.empty((0, 8))
        # Input: [T1, T2, T3, T4]
        self.U = np.empty((0, 4))

    def set_state(self, pos: np.ndarray = np.zeros(8)):
        """
        Set initial state
        :param pos: [x, y, z, ψ, u, v, w, r]
        """
        self.X = pos.reshape(1, -1)
        return self.X

    def generate_input(self, horizon: int = 100):
        """
        Generate random thruster inputs using normal distribution
        :param horizon: Number of control steps
        """
        self.U = np.random.normal(0, self.max_thrust/3, (horizon, 4))
        self.U = np.clip(self.U, -self.max_thrust, self.max_thrust)
        return self.U

    def get_jacobians(self):
        # Extract state
        ψ = self.X[0, 3]
        u, v, w, r = self.X[0, 4:]
        m = self.mass
        Iz = self.inertia_z
        Xu = self.hydro['X_u']
        Xudot = self.hydro['X_udot']
        Yv = self.hydro['Y_v']
        Yvdot = self.hydro['Y_vdot']
        Zw = self.hydro['Z_w']
        Zwdot = self.hydro['Z_wdot']
        Nr = self.hydro['N_r']
        Nrdot = self.hydro.get('N_rdot', 0.0)  # Use 0 if not defined

        # Precompute denominators
        mxu = m - Xudot
        myv = m - Yvdot
        mzw = m - Zwdot
        inz = Iz - Nrdot

        dt = self.dt

        # Initialize A as identity
        A = np.eye(8)

        # Kinematic part (x, y, z, psi)
        A[0, 4] = dt * np.cos(ψ)  # ∂x/∂u
        A[0, 5] = -dt * np.sin(ψ) # ∂x/∂v
        A[1, 4] = dt * np.sin(ψ)  # ∂y/∂u
        A[1, 5] = dt * np.cos(ψ)  # ∂y/∂v
        A[2, 6] = dt              # ∂z/∂w
        A[3, 7] = dt              # ∂ψ/∂r

        # Dynamic part (u, v, w, r)
        # ∂(dot u)/∂v and ∂(dot u)/∂r due to vr
        A[4, 5] = dt * (m - Yvdot) * r / mxu
        A[4, 7] = dt * (m - Yvdot) * v / mxu
        # ∂(dot u)/∂u due to -X_u u
        A[4, 4] = 1 - dt * Xu / mxu

        # ∂(dot v)/∂u and ∂(dot v)/∂r due to -ur
        A[5, 4] = -dt * (m - Xudot) * r / myv
        A[5, 7] = -dt * (m - Xudot) * u / myv
        # ∂(dot v)/∂v due to -Y_v v
        A[5, 5] = 1 - dt * Yv / myv

        # ∂(dot w)/∂w due to -Z_w w
        A[6, 6] = 1 - dt * Zw / mzw

        # ∂(dot r)/∂u and ∂(dot r)/∂v due to - (Xudot-Yvdot)uv
        A[7, 4] = -dt * (Xudot - Yvdot) * v / inz
        A[7, 5] = -dt * (Xudot - Yvdot) * u / inz
        # ∂(dot r)/∂r due to -N_r r
        A[7, 7] = 1 - dt * Nr / inz

        # Input Jacobian (B) remains as before
        B = np.zeros((8, 4))
        M_inv = np.diag([1/mxu, 1/myv, 1/mzw, 1/inz])
        B[4:8, :] = dt * (M_inv @ self.B)

        return A, B
    
    def predict(self, horizon: int = 100):
        """
        Predict state trajectory using current inputs
        :param horizon: Prediction horizon
        """
        if self.U.shape[0] < horizon:
            self.generate_input(horizon)
        
        A, B = self.get_jacobians()
        X_pred = np.zeros((horizon+1, 8))
        X_pred[0] = self.X[0]
        
        for t in range(horizon):
            X_pred[t+1] = A @ X_pred[t] + B @ self.U[t]
        
        return X_pred

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _ in range(100):
        # Initialize robot with default parameters
        robot = underwaterRobotModel(dt=0.1)
        robot.set_state(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        robot.generate_input(horizon=100)
        trajectory = robot.predict()
        # Plotting the trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Robot Trajectory')

    # Visualization
    ax.set_title('Underwater Robot Trajectory Prediction')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.tight_layout()
    plt.show()
