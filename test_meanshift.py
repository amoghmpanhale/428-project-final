# test_meanshift.py

import cv2
import numpy as np
import time
from src.mean_shift import MeanShiftTracker
from utils import *

# Paths to data and outputs
video_path = 'data/animaltrack/deer_2.mp4'
annotation_path = 'data/animaltrack/deer_2.txt'
output_video_path = 'meanshift_tracked_video.mp4'
iou_plot_path = 'meanshift_iou.png'
fps_plot_path = 'meanshift_fps.png'

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

# Loop over the frames in thew video
while True:
    start_time = time.time() 
    
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker and get the new current position
    current_x, current_y = tracker.update(frame)
    current_w, current_h = tracker.window_size

    prediction_box = (current_x, current_y, current_w, current_h)

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
plot_metrics(ious, fps_values, "Meanshift")



