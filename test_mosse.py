# test_mosse.py

import cv2
import numpy as np
import time
from utils import *

# Paths to data and outputs
video_path = 'data/animaltrack/penguin_5.mp4'
annotation_path = 'data/animaltrack/penguin_5.txt'
output_video_path = 'mosse_tracked_video.mp4'
iou_plot_path = 'mosse_iou.png'
fps_plot_path = 'mosse_fps.png'

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

# Initialize the MOSSE tracker
tracker = cv2.legacy.TrackerMOSSE_create()
success = tracker.init(frame, initial_box)

# Loop over the frames in thew video
while True:
    start_time = time.time() 
    
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker and get the new current position
    success, current_box = tracker.update(frame)
    
    if not success:
        print(f"Tracking failed at frame {frame_count}")
        continue

    current_x, current_y, current_w, current_h = int(current_box[0]), int(current_box[1]), int(current_box[2]), int(current_box[3])

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
plot_metrics(ious, fps_values, "MOSSE")
