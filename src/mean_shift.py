# mean_shift.py
'''
This file contains a class for the mean shift tracker for the project. This implementation was derived from the Comaniciu paper.
some referneces for code logic come from this github project: https://github.com/Blarc/mean-shift-tracking
'''

import numpy as np
import cv2

class MeanShiftTracker:
    def __init__(self, kernel='normal', num_bins= 64, bandwidth=30, max_iter=50, eps=0.0001):
        '''
        The initialization portion of the tracker class. The kind of kernel does affect performance. 
        For the purpose of this project we are using the epanechnikov and normal kernels.
        '''
        self.num_bins = num_bins # The number of bins for the grayscale histogram from the target model, m
        self.h = bandwidth # The kernel bandwidth, h
        self.max_iter = max_iter # The maximum number of thresholds for the iterative loop
        self.eps = eps # The convergence threshold for the iterative shift vector calculation
        self.kernel = kernel # The kernel used for weighted mean calculation

        self.q = None # The target model
        self.model_coords = None # The normalized pixel values within the initial window
        self.model_bins = None # The indices of the bins in the model pixels as per the input num bins
        self.current_center = None
        self.initial_center = None # Initial center of the bounding box region pass in init()
        self.window_size = None # The width and height of the windowed region to be tracked
        self.K = None

    def kernel_profile(self, u):
        '''
        Computes the kernel weights for the given coordinates as per the implementation in the paper. 
        Constants are ignored as they don't change and get cancelled out.
        epanechnikov: max(1 - u, 0)
        Normal: exp(-u/2)
        '''
        if self.kernel == 'epanechnikov':
            return np.maximum(1.0 - u, 0.0)
        else: # Since we only have two cases, want to prevent code that falls through
            return np.exp(-0.5 * u)

    def init(self, frame, roi):
        '''
        Initialization function that actually takes the frame and a bounding box for the initial target model initialization.
        This portion is extracted from sections 3 and 4 of the Comaniciu paper.
        The responsibility of this function is to calculate the target model of the initial bounding box region q with the number of bins preset
        and the values of the colors normalized.
        '''
        x, y, w, h = roi

        # Conver the frame to grayscale if it isn't already in it
        gray_frame = frame
        if frame.ndim == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target_patch = gray_frame[y:y + h, x:x + w].astype(np.float32)

        # Create a flattened gird out of the coordinates
        x_list = np.linspace(-1, 1, w)
        y_list = np.linspace(-1, 1, h)
        x_vec, y_vec = np.meshgrid(x_list, y_list)
        coordinates = np.vstack([x_vec.reshape(-1), y_vec.reshape(-1)]).T # Makes an Nx2 matrix

        # Initialize u and k as per the paper
        u = np.sum(coordinates**2, axis=1) # the color coordinates u as seen in section 4
        k = self.kernel_profile(u)

        # Computing function b of the colors as per section 4.1
        pixels = target_patch.reshape(-1) # flatten the initial region
        bins = np.floor(pixels * self.num_bins / 256.0).astype(int) # Scale the pixel values to match uint8 format for the histogram based on the number of bins m
        bins = np.clip(bins, 0, self.num_bins - 1) # Clip the values to ensure nothing falls outside of the histogram range

        # Apply the weights to the histogram using the kernel profile k and normalize to get q
        q = np.bincount(bins, weights=k, minlength=self.num_bins).astype(np.float32)
        q_sum = q.sum() 
        if q_sum > 0: # Normalize the target model
            self.q = q / q_sum
        else:
            self.q = np.ones(self.num_bins, dtype=np.float32) / self.num_bins

        # Store the calculated target model
        self.model_coords = coordinates
        self.model_bins = bins
        self.initial_center = np.array([x + w/2, y + h/2], dtype=np.float32)
        self.window_size = (w, h)
        self.current_center = self.initial_center.copy()
        self.K = k

    def get_candidate_histogram(self, frame, roi_center):
        '''
        This function calculates the histogram of candidate regions given the next frame from the center coordinate provided.
        This implementation is done as per the distance minization portion of the Comaniciu paper (4.2) 
        '''
        x_center, y_center = roi_center.astype(int)
        H, W = frame.shape[:2]

        # Convert the frame to grayscale if it isn't already in it
        gray_frame = frame
        if frame.ndim == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract the patch from the given frame as per equation 21
        w_0, h_0 = self.window_size
        patch = cv2.getRectSubPix(gray_frame, (w_0, h_0), tuple(roi_center)).astype(np.float32)

        # # Create a flattened gird out of the coordinates
        # x_list = np.linspace(-1, 1, pw)
        # y_list = np.linspace(-1, 1, ph)
        # x_vec, y_vec = np.meshgrid(x_list, y_list)
        # coordinates = np.vstack([x_vec.reshape(-1), y_vec.reshape(-1)]).T # Makes an Nx2 matrix

        # # Initialize u and k
        # u = np.sum(coordinates**2, axis=1)
        # k = self.kernel_profile(u)

        pixels = patch.reshape(-1)
        bins = np.floor(pixels * self.num_bins / 256.0).astype(int) # Scale the pixel values to match uint8 format for the histogram based on the number of bins m
        bins = np.clip(bins, 0, self.num_bins - 1) # Clip the values to ensure nothing falls outside of the histogram range

        # Calculate p in a similar fashion to q and normalize
        p = np.bincount(bins, weights=self.K, minlength=self.num_bins).astype(np.float32)
        s = p.sum()
        return p / (s + 1e-8)

    def compute_weights(self, p):
        '''
        This function calculates w_i as is required by step 2 of the algorithm in section 4.2 using equation 25
        '''
        return np.sqrt(self.q[self.model_bins] / (p[self.model_bins] + 1e-8))

    def calculate_bhattacharyya(self, p):
        '''
        Calculates the Bhattacharyya coefficient based on the formula on page 4 of the paper
        '''
        return np.sum(np.sqrt(p * self.q))

    def update(self, frame):
        '''
        Main update function to perform Bhattacharyya minimization as seen in section 4.2 of the paper
        returns the new center of the bounding box 
        '''
        y_0 = self.current_center.copy()
        for i in range(self.max_iter):
            # 1) Initialize the location of the current frame
            p_hat = self.get_candidate_histogram(frame, y_0)
            rho = self.calculate_bhattacharyya(p_hat)

            # 2) Derive the weights according to 25
            w_i = self.compute_weights(p_hat)

            # 3) Derive the new location of the target based on the mena shift vector
            K = self.K
            y_norm = (np.sum(self.model_coords * (w_i[:,None] * K[:,None]), axis=0) / (np.sum(w_i * K) + 1e-8))
            w_win, h_win = self.window_size
            y_1_image = y_0 + np.array([y_norm[0] * w_win/2.0, y_norm[1] * h_win/2.0], dtype=np.float32)

            # 4) Iteratively update using the Bhattacharyya
            p_hat_1 = self.get_candidate_histogram(frame, y_1_image)
            rho_1 = self.calculate_bhattacharyya(p_hat_1)
            num_iters = 0
            while rho_1 < rho and num_iters <= self.max_iter:
                y_1_image = 0.5 * (y_0 + y_1_image)
                p_hat_1 = self.get_candidate_histogram(frame, y_1_image)
                rho_1 = self.calculate_bhattacharyya(p_hat_1)
                num_iters += 1
            
            # 5) Stop when there is convergence based on epsilon
            if np.linalg.norm(y_1_image - y_0) < self.eps:
                y_0 = y_1_image
                break
            y_0 = y_1_image
        
        self.current_center = y_0

        print(tuple(y_0.astype(int)))

        return tuple(y_0.astype(int))









