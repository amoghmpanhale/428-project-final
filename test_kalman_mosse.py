# test_kalman_mosse.py

import cv2
import numpy as np
import time
from src.kl_discrete import DiscreteKalmanFilter
from utils import *

# Paths to data and outputs
video_path = 'data/animaltrack/penguin_5.mp4'
annotation_path = 'data/animaltrack/penguin_5_sliding.txt'
output_video_path = 'kalman_mosse_tracked_video.mp4'
iou_plot_path = 'kalman_mosse_iou.png'
fps_plot_path = 'kalman_mosse_fps.png'

ground_truth = load_ground_truth(annotation_path)
cap, out, W, H, fps = setup_video(video_path, output_video_path)

# Fetch the initial bounding box
x, y, w, h = ground_truth[0]
initial_box = (x, y, w, h)

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

# Initialize the MOSSE tracker
tracker = cv2.legacy.TrackerMOSSE_create()
success = tracker.init(frame, initial_box)

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
    success, current_box = tracker.update(frame)
    
    if success:
        current_x, current_y, current_w, current_h = int(current_box[0]), int(current_box[1]), int(current_box[2]), int(current_box[3])
        measurement = [current_x + current_w/2, current_y + current_h/2]
    
        # Update Kalman filter with new measurement
        kalman.predict()
        updated_state = kalman.update(measurement)
    else:
        print(f"Tracking failed at frame {frame_count}")
        updated_state = kalman.predict()

    # Get Kalman position
    kalman_x, kalman_y = updated_state[0], updated_state[1]
    kalman_box = ((kalman_x - w/2), (kalman_y - h/2), w, h)

    # Get the ground truth for this frame and calculate the iou for it and draw it on the frame
    gt_box = ground_truth[frame_count]
    ious.append(compute_iou(gt_box, kalman_box))
    frame = draw_boxes(frame, kalman_box, gt_box)
    
    # Draw the MOSSE box in blue
    if success:
        mosse_box = (current_x, current_y, current_w, current_h)
        cv2.rectangle(frame, (current_x, current_y), 
                    (current_x + current_w, current_y + current_h), (255, 0, 0), 2)

    # Save the frame and calculate FPS
    out.write(frame)
    fps_values.append(1.0 / (time.time() - start_time))
    frame_count += 1

# Cleanup and display results
cap.release()
out.release()
print_results(ious, fps_values, frame_count, ground_truth)
plot_metrics(ious, fps_values, "Kalman_MOSSE")
