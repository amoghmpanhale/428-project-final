# utils.py

'''
This script contains utility functions used in the tracker tests that are just common things done. 
Some reference was taken to code from the labs and assignments.
The AI tool Claude was also used to detect typos and provide comments for documentation.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#region compute_iou
def compute_iou(boxA, boxB):
    '''
    A function to calculate the intersection over union of two bounding boxes
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / (boxAArea + boxBArea - interArea + 1e-8)
#endregion

#region load_ground_truth
def load_ground_truth(annotation_path):
    '''
    Load ground truth bounding boxes from a txt file
    '''
    ground_truth_boxes = []
    with open(annotation_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                x, y, w, h = map(int, line.strip().split())
                ground_truth_boxes.append((x, y, w, h))
    return ground_truth_boxes
#endregion

#region setup_video
def setup_video(video_path, output_video_path):
    '''
    Setup video capture and writer
    '''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    return cap, out, W, H, fps
#endregion

#region draw_boxes
def draw_boxes(frame, pred_box, gt_box=None):
    '''
    Draw prediction and ground truth bounding boxes on frame
    '''
    # Green for prediction
    x1, y1, w, h = pred_box
    pt1 = (int(x1), int(y1))
    pt2 = (int(x1 + w), int(y1 + h))
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Red for ground truth
    if gt_box is not None:
        gt_x, gt_y, gt_w, gt_h = gt_box
        gt_pt1 = (gt_x, gt_y)
        gt_pt2 = (gt_x + gt_w, gt_y + gt_h)
        cv2.rectangle(frame, gt_pt1, gt_pt2, (0, 0, 255), 2)
    
    return frame
#endregion

#region plot_metrics
def plot_metrics(ious, fps_values, tracker_type):
    '''
    Plot IoU and FPS metrics
    '''
    # Plot IoU
    plt.figure(figsize=(10, 5))
    plt.plot(ious, label="IoU")
    plt.title(f"IoU Over Time ({tracker_type})")
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{tracker_type}_iou.png')
    plt.close()

    # Plot FPS
    plt.figure(figsize=(10, 5))
    plt.plot(fps_values, label="FPS")
    plt.title(f"FPS Over Time ({tracker_type})")
    plt.xlabel("Frame")
    plt.ylabel("Frames Per Second")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{tracker_type}_fps.png')
    plt.close()
#endregion

#region print_results
def print_results(ious, fps_values, frame_count, ground_truth_boxes):
    '''
    Print mean IoU, FPS and frame information
    '''
    mean_iou = np.mean(ious) if ious else 0
    mean_fps = np.mean(fps_values) if fps_values else 0
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean FPS: {mean_fps:.2f}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total ground truth boxes: {len(ground_truth_boxes)}")
#endregion