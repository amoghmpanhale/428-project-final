# test_iclk.py

import cv2
import numpy as np
import time
from src.mean_shift import MeanShiftTracker
from src.iclk import InverseCompositionalLucasKanadeTracker
from utils import *

# Paths to data and outputs
video_path = 'data/animaltrack/penguin_5.mp4'
annotation_path = 'data/animaltrack/penguin_5.txt'
output_video_path = 'iclk_tracked_video.mp4'
iou_plot_path = 'iclk_iou.png'
fps_plot_path = 'iclk_fps.png'

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
first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define the template for the tracker
template = first_frame_grayscale[y : y + h//2, x : x + w//2] 

# Initialize the variables for tracking
ious = []
fps_values = []
frame_count = 0
current_x, current_y = x, y

# Initialize the inv comp lucas kanade tracker
tracker = InverseCompositionalLucasKanadeTracker(template)

# Loop over the frames in thew video
while True:
    start_time = time.time() 
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    y = max(0, int(current_y)) # Making sure y isn't negative or a float
    y1 = min(gray.shape[0], y + h) # Making sure the bottom y doesn't go past the frame boundaries
    x = max(0, int(current_x)) # Making sure x isn't negative or a float
    x1 = min(gray.shape[1], x + w) # Making sure the x doesn't go outside the frame
    roi_frame = gray[y : y1, x : x1]

    # Get the motion vector from the tracker
    motion_vector = tracker.track(roi_frame)
    current_x += motion_vector[0]
    current_y += motion_vector[1]

    prediction_box = (current_x, current_y, w, h)

    # Get the ground truth for this frame and calculate the iou for it and draw it on the frame
    gt_box = ground_truth[frame_count]
    ious.append(compute_iou(gt_box, prediction_box))
    frame = draw_boxes(frame, prediction_box, gt_box)

    # Save the frame and calculate FPS
    out.write(frame)
    fps_values.append(1.0 / (time.time() - start_time))
    frame_count += 1

# Cleanup and display results
cap.release()
out.release()
print_results(ious, fps_values, frame_count, ground_truth)
plot_metrics(ious, fps_values, "ICLK")

