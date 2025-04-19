# test_kalman_meanshift.py

import cv2
import numpy as np
import time
from src.mean_shift import MeanShiftTracker
from src.kl_discrete import DiscreteKalmanFilter
from utils import *

# Paths to data and outputs
video_path = 'data/animaltrack/penguin_5.mp4'
annotation_path = 'data/animaltrack/penguin_5_sliding.txt'
output_video_path = 'kalman_meanshift_tracked_video.mp4'
iou_plot_path = 'kalman_meanshift_iou.png'
fps_plot_path = 'kalman_meanshift_fps.png'

ground_truth = load_ground_truth(annotation_path)
cap, out, W, H, fps = setup_video(video_path, output_video_path)


# Fetch the initial bounding box
x, y, w, h = ground_truth[0]
initial_box = (x, y, w // 2, h // 2)

# Read the first frame in the video
ret, frame = cap.read()
if not ret:
    print("Can't read video")
    cap.release()
    exit()

# Initialize the variables for tracking
ious = []
fps_values = []
frame_count = 0

# Initialize a mean shift tracker object
tracker = MeanShiftTracker()

# Initialize the mean shift tracker's target model
tracker.init(frame, initial_box)

# Initialize Kalman filter with the initial state
dt = 1.0 / fps # The frequency based on the frames per second
kalman = DiscreteKalmanFilter([x + w/2, y + h/2, 0, 0], dt) 

# Loop over the frames in thew video
while True:
    start_time = time.time() 
    
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker and get the new current position
    current_x, current_y = tracker.update(frame)
    current_w, current_h = tracker.window_size
    measurement = [current_x + current_w/2, current_y + current_h/2]

    # Update Kalman filter with new measurement
    kalman.predict()
    updated_state = kalman.update(measurement)

    # Get Kalman position
    kalman_x, kalman_y = updated_state[0], updated_state[1]
    kalman_box = ((kalman_x - w/2), (kalman_y - h/2), w, h)

    # Get the ground truth for this frame and calculate the iou for it and draw it on the frame
    gt_box = ground_truth[frame_count]
    ious.append(compute_iou(gt_box, kalman_box))
    frame = draw_boxes(frame, kalman_box, gt_box)
    
    # Draw the MeanShift box in blue
    cv2.rectangle(frame, (int(current_x), int(current_y)), 
                  (int(current_x + current_w), int(current_y + current_h)), (255, 0, 0), 2)

    # Save the frame and calculate FPS
    out.write(frame)
    fps_values.append(1.0 / (time.time() - start_time))
    frame_count += 1

# Cleanup and display results
cap.release()
out.release()
print_results(ious, fps_values, frame_count, ground_truth)
plot_metrics(ious, fps_values, "Kalman MeanShift")