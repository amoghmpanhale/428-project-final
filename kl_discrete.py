# kl_discrete.py

'''
This script implements the discrete Kalman Filter as per the Welch paper. 
This case assumes a constant velocity model of the motion of the tracked object.
Some of the understanding and terminology is also sourced from the matlab youtube series on the kalman filter: https://www.youtube.com/watch?v=mwn8xhgNpFY&list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr&ab_channel=MATLAB
Some of the conversion from math to code was also referenced from this github repository: https://github.com/zziz/kalman-filter
'''

import numpy as np

class DiscreteKalmanFilter:
    '''
    This class represents the discrete implementation of the Kalman filter. 
    The state space model of this case assumes constant velocity.
    State vector: [x, y, dx, dy]
    Measurements: [x, y] coordinates for the center of the tracked object
    '''

    def __init__(self, initial_state, dt, process_noise=0.5, measurement_noise=0.5):
        '''
        Initializes the filter using a state and dt passed. Implementation is based on the Welch paper.
        '''

        # Initialize the state 
        self.x_hat = np.array(initial_state, dtype=np.float32)
        self.dt = dt

        # Initialize the state transition matrix A for simple motion
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        # Initialize the measurement matrix H 
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Initialize the process noise covariance Q
        self.Q = process_noise * np.eye(4)

        # Initialize the measurement noise covariance R
        self.R = measurement_noise * np.eye(2)

        # Initialize the error covariance P
        self.P = np.eye(4)

    def predict(self):
        '''
        This function represents the prediction step of the filter. It is based on the equations 1.9 and 1.10 in the Welch paper
        '''
        # Equation 1.9
        self.x_hat = self.A @ self.x_hat

        # Equation 1.10
        self.P = (self.A @ self.P @ self.A.T) + self.Q

        return self.x_hat.copy()

    def update(self, measurement):
        '''
        This function represents the update step of the filter. It  is based on the equations 1.11, 1.12, and 1.13
        '''
        # Calculate kalman gain K
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        # Calculate the state update
        self.x_hat = self.x_hat + (K @ (np.array(measurement, dtype=np.float32) - (self.H @ self.x)))

        # Calculate the error covariance P
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x_hat.copy()
