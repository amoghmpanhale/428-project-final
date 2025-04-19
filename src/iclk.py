# iclk.py

'''
This script contains a class for the implementation of the translational Inverse Compositional Lucas Kanade Tracker. This implementation is based on the Baker and Simon Paper.
'''

import numpy as np
import cv2

class InverseCompositionalLucasKanadeTracker:
    def __init__(self, template, threshold=0.00001, max_iters=200):
        '''
        The initialization of the tracker. This step involves precomputation of the jacobian and hessian of the template. It comes from pages 12 to 13 of the paper.
        '''

        self.T = template.astype(np.float32)
        self.height, self.width = self.T.shape
        self.threshold = threshold
        self.max_iters = max_iters

        # Precomputing steps 3-6 in the Inverse Compositional Algorithm
        # 3) Evaluate the gradient delta T of the template T
        self.gradient_x = cv2.Sobel(self.T, cv2.CV_32F, 1, 0, ksize=3)
        self.gradient_y = cv2.Sobel(self.T, cv2.CV_32F, 0, 1, ksize=3)

        # For the translational case, there is no need to directly compute a Jacobian. 

        # 5) Compute the sttepest descent images delta T dW/dp
        self.steepest_descent = np.zeros((self.height * self.width, 2), dtype=np.float32)
        self.steepest_descent[:, 0] = self.gradient_x.reshape(-1) 
        self.steepest_descent[:, 1] = self.gradient_y.reshape(-1)

        # 6) Compute the Hessian matrix using the equation 38 which is just a summation of the square of the steepest descent
        self.H = self.steepest_descent.T @ self.steepest_descent
        self.H_inverse = np.linalg.inv(self.H)

    def translational_warp(self, p):
        '''
        Create and apply an translational warp based on a parameter vector p
        '''
        return np.array([[1, 0, p[0]], [0, 1, p[1]]], dtype=np.float32)

    def track(self, frame):
        '''
        Calculates the updated parameter matrix p given a new frame. This is an extension of the algorithm applied in the init from the same paper.
        '''
        # Initialize the parameter matrix p
        p = np.zeros(2, dtype=np.float32)
        frame_float = frame.astype(np.float32)

        for i in range(self.max_iters):
            # 1) Warp I with W(x;p) to compute I(W(x;p))
            warp_matrix = self.translational_warp(p)
            I_W = cv2.warpAffine(frame_float, warp_matrix, (self.width, self.height))

            # 2) Compute the Error Image I(W(x;p)) - T(x)
            error = (I_W - self.T).reshape(-1)

            # 7) Compute the steepest descent error
            sd_error = self.steepest_descent.T @ error

            # 8) Compute the parameter update dp using the formula 37
            dp = self.H_inverse @ sd_error

            # Check to see if the loop has converged
            if np.linalg.norm(dp) < self.threshold:
                break

            # 9) Update the warp W(x;p)
            p[0] -= dp[0]
            p[1] -= dp[1]
        
        print(p)

        return p