import streamlit as st
import cv2
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Wildlife Tracking with Classical Computer Vision",
    page_icon="🦓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add Open Graph meta tags for LinkedIn preview
st.markdown("""
    <meta property="og:title" content="Wildlife Tracking with Classical Computer Vision" />
    <meta property="og:description" content="Interactive demo comparing ICLK, MOSSE, and Mean Shift tracking algorithms for wildlife monitoring. Built with Python, OpenCV, and Streamlit." />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="https://amoghmpanhale-428-project-final.streamlit.app" />
    <meta property="og:image" content="https://raw.githubusercontent.com/amoghmpanhale/428-project-final/main/outputs/output_deer/mosse_fps.png" />
    <meta name="description" content="Wildlife tracking project comparing classical CV algorithms - MOSSE (60+ FPS), Lucas-Kanade, Mean Shift, with Kalman filtering." />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="Wildlife Tracking with Classical Computer Vision" />
    <meta name="twitter:description" content="Interactive demo comparing tracking algorithms for wildlife monitoring" />
    <meta name="twitter:image" content="https://raw.githubusercontent.com/amoghmpanhale/428-project-final/main/outputs/output_deer/mosse_fps.png" />
    """, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .algorithm-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .metric-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">Wildlife Tracking with Classical Computer Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparative Analysis of Object Tracking Algorithms for Animal Monitoring</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Algorithm Demonstrations", "Technical Details", "Performance Comparison"])

# Overview Page
if page == "Overview":
    st.header("Project Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### About This Project
        This project implements and compares classical computer vision tracking algorithms for wildlife monitoring applications.
        It was developed as part of CMPUT 428 at the University of Alberta.

        ### Key Features
        - **Multiple Tracking Algorithms**: ICLK, MOSSE, Mean Shift
        - **Kalman Filter Integration**: Enhanced tracking with predictive filtering
        - **Performance Metrics**: IoU and FPS tracking for quantitative analysis
        - **Real-world Dataset**: Tested on AnimalTrack and Zebra datasets

        ### Motivation
        Traditional wildlife monitoring requires manual video analysis, which is time-consuming and error-prone.
        This project explores classical computer vision techniques as a foundation for automated animal tracking,
        comparing their effectiveness before scaling to more complex deep learning approaches.
        """)

    with col2:
        st.markdown("""
        ### Algorithms Implemented
        - **ICLK**: Inverse Compositional Lucas-Kanade
        - **MOSSE**: Minimum Output Sum of Squared Error
        - **Mean Shift**: Gradient-based tracking
        - **Kalman Filter**: Discrete predictive filter

        ### Technologies
        - Python 3.11
        - OpenCV 4.11
        - NumPy
        - Matplotlib
        """)

    st.markdown("---")

    st.subheader("Project Structure")
    st.code("""
    src/
    ├── iclk.py              # Inverse Compositional LK implementation
    ├── mean_shift.py        # Mean Shift tracker
    └── kl_discrete.py       # Kalman Filter

    data/
    ├── animaltrack/         # AnimalTrack dataset
    └── zebra_dataset/       # Zebra tracking data

    outputs/
    ├── output_deer/         # Deer tracking results
    └── output_zebra/        # Zebra tracking results
    """, language="text")

# Algorithm Demonstrations Page
elif page == "Algorithm Demonstrations":
    st.header("Algorithm Demonstrations")

    st.markdown("""
    Select an algorithm and dataset to see the tracking results and performance metrics.
    The videos show the tracker's predictions (green) compared to ground truth annotations (blue).
    """)

    # Selection controls
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox(
            "Select Algorithm",
            ["MOSSE", "ICLK (Lucas-Kanade)", "Mean Shift", "Kalman + MOSSE", "Kalman + ICLK", "Kalman + Mean Shift"]
        )

    with col2:
        dataset = st.selectbox(
            "Select Dataset",
            ["Deer", "Zebra"]
        )

    # Map selections to file paths
    algo_map = {
        "MOSSE": "mosse",
        "ICLK (Lucas-Kanade)": "iclk",
        "Mean Shift": "meanshift",
        "Kalman + MOSSE": "kalman_mosse",
        "Kalman + ICLK": "kalman_iclk",
        "Kalman + Mean Shift": "kalman_meanshift"
    }

    dataset_map = {
        "Deer": "deer",
        "Zebra": "zebra"
    }

    algo_key = algo_map[algorithm]
    dataset_key = dataset_map[dataset]

    # Check for Kalman filter
    if "Kalman" in algorithm:
        output_dir = f"outputs/output_{dataset_key}_kalman_{algo_key.split('_')[-1]}"
        video_name = "kalman_tracked_video.mp4"
        iou_plot = "kalman_iou.png"
        fps_plot = "kalman_fps.png"
    else:
        output_dir = f"outputs/output_{dataset_key}"
        video_name = f"{algo_key}_tracked_video.mp4"
        iou_plot = f"{algo_key}_iou.png"
        fps_plot = f"{algo_key}_fps.png"

    video_path = Path(output_dir) / video_name
    iou_path = Path(output_dir) / iou_plot
    fps_path = Path(output_dir) / fps_plot

    # Display results if they exist
    if video_path.exists():
        st.subheader(f"{algorithm} Tracking on {dataset} Dataset")

        # Display video
        st.video(str(video_path))

        st.markdown("---")

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            if fps_path.exists():
                st.subheader("Frames Per Second (FPS)")
                st.image(str(fps_path), width="stretch")
                st.caption("Higher FPS indicates faster processing and real-time capability")

        with col2:
            if iou_path.exists():
                st.subheader("Tracking Accuracy")
                st.image(str(iou_path), width="stretch")
                st.caption("Overlap between predicted and ground truth bounding boxes over time")
    else:
        st.warning(f"Results not available for {algorithm} on {dataset} dataset. Available combinations may be limited to deer dataset.")
        st.info("Try selecting 'Deer' as the dataset to see available results.")

# Technical Details Page
elif page == "Technical Details":
    st.header("Technical Details")

    tab1, tab2, tab3, tab4 = st.tabs(["ICLK Tracker", "MOSSE Tracker", "Mean Shift Tracker", "Kalman Filter"])

    with tab1:
        st.subheader("Inverse Compositional Lucas-Kanade (ICLK) Tracker")

        st.markdown("""
        ### Algorithm Overview
        The ICLK tracker is based on the Baker & Matthews (2004) paper. It's an efficient optical flow-based tracking algorithm
        that inverts the traditional Lucas-Kanade approach for computational efficiency.

        ### Key Innovations
        - **Precomputation**: Gradient and Hessian computed once on template
        - **Inverse Composition**: Warps the template instead of the image
        - **Efficiency**: Reduces per-frame computation significantly

        ### Implementation Highlights
        """)

        st.code("""
class InverseCompositionalLucasKanadeTracker:
    def __init__(self, template, threshold=0.0001, max_iters=30):
        # Precompute gradients using Sobel operator
        self.gradient_x = cv2.Sobel(self.T, cv2.CV_32F, 1, 0, ksize=3)
        self.gradient_y = cv2.Sobel(self.T, cv2.CV_32F, 0, 1, ksize=3)

        # Compute steepest descent images
        self.steepest_descent = np.zeros((self.height * self.width, 2))
        self.steepest_descent[:, 0] = self.gradient_x.reshape(-1)
        self.steepest_descent[:, 1] = self.gradient_y.reshape(-1)

        # Precompute Hessian inverse (H^-1)
        self.H = self.steepest_descent.T @ self.steepest_descent
        self.H_inverse = np.linalg.inv(self.H)

    def track(self, frame):
        # Iterative refinement until convergence
        for i in range(self.max_iters):
            # Warp, compute error, update parameters
            warp_matrix = self.translational_warp(p)
            I_W = cv2.warpAffine(frame_float, warp_matrix, (self.width, self.height))
            error = (I_W - self.T).reshape(-1)

            # Compute parameter update
            dp = self.H_inverse @ (self.steepest_descent.T @ error)

            if np.linalg.norm(dp) < self.threshold:
                break

            p -= dp
        return p
        """, language="python")

        st.markdown("""
        ### Strengths & Limitations
        - **Strengths**: Fast, efficient, works well on textured objects
        - **Limitations**: Sensitive to large motions, requires good initialization
        - **Best Use Case**: Tracking animals with distinct patterns in controlled environments
        """)

    with tab2:
        st.subheader("MOSSE (Minimum Output Sum of Squared Error) Tracker")

        st.markdown("""
        ### Algorithm Overview
        MOSSE is a correlation filter-based tracker (Bolme et al., 2010) that learns a filter to produce
        a peak response at the object's location.

        ### Key Features
        - **Adaptive**: Updates filter online during tracking
        - **Fast**: Uses FFT for efficient correlation
        - **Robust**: Handles appearance changes and partial occlusions

        ### Why MOSSE Performs Best
        Based on our experiments:
        - **Highest IoU**: Better overlap with ground truth
        - **Highest FPS**: Real-time performance (60+ FPS)
        - **Robustness**: Handles animal movement patterns well

        ### Implementation
        Uses OpenCV's legacy tracker implementation:
        """)

        st.code("""
# Initialize MOSSE tracker
tracker = cv2.legacy.TrackerMOSSE_create()
success = tracker.init(frame, initial_box)

# Track across frames
while True:
    ret, frame = cap.read()
    success, current_box = tracker.update(frame)

    # Compute IoU with ground truth
    iou = compute_iou(ground_truth_box, current_box)
        """, language="python")

        st.markdown("""
        ### Performance Metrics
        In our deer tracking experiments:
        - Average FPS: 60-80 (real-time performance)
        - High tracking accuracy on structured movements
        - Success rate: >90% frame-to-frame
        """)

    with tab3:
        st.subheader("Mean Shift Tracker")

        st.markdown("""
        ### Algorithm Overview
        Mean Shift (Comaniciu et al., 2000) is a gradient ascent algorithm that finds modes in probability distributions.
        For tracking, it finds the region with the highest similarity to the target's color histogram.

        ### How It Works
        1. Compute color histogram of target
        2. For each frame, search for region with similar histogram
        3. Shift bounding box toward highest similarity
        4. Iterate until convergence

        ### Challenges with Wildlife
        - **Groups**: Struggles when animals travel in groups (similar appearances)
        - **Background**: Confusion with similarly-colored backgrounds
        - **Occlusions**: Difficulty recovering from partial occlusions

        ### Our Findings
        From the project experiments:
        > "Initially, it was thought that the mean shift tracker would allow for animal patterns to be
        > easily tracked but animals traveling in groups tend to confuse the tracker to the point of
        > negligible tracking."

        This motivated exploration of correlation filter approaches like MOSSE.
        """)

    with tab4:
        st.subheader("Discrete Kalman Filter")

        st.markdown("""
        ### Algorithm Overview
        The Kalman Filter is a recursive state estimator that combines noisy measurements with predictions
        to produce optimal estimates of object state.

        ### Integration with Trackers
        The Kalman filter enhances base trackers by:
        1. **Prediction**: Estimates next position based on motion model
        2. **Correction**: Updates estimate with tracker measurement
        3. **Smoothing**: Reduces jitter and handles brief occlusions

        ### State Vector
        For 2D tracking with velocity:
        - Position: (x, y)
        - Velocity: (vx, vy)

        ### Results
        From experiments:
        > "The performance of the Discrete Kalman Filter is definitely dependent on the performance
        > of the tracker that trains it. Tests with it have shown some minor increases in IoU."

        ### Key Insight
        Kalman filtering provides marginal improvements when the base tracker is already robust (like MOSSE),
        but can help smooth trajectories and handle brief tracking failures.
        """)

# Performance Comparison Page
elif page == "Performance Comparison":
    st.header("Performance Comparison")

    st.markdown("""
    ### Comparative Analysis
    Based on extensive testing on the deer and zebra datasets, here's how the algorithms compare:
    """)

    # Create comparison table
    st.subheader("Algorithm Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("### MOSSE")
        st.markdown("**Best Overall**")
        st.markdown("Accuracy: ⭐⭐⭐⭐⭐")
        st.markdown("FPS: ⭐⭐⭐⭐⭐")
        st.markdown("Robustness: ⭐⭐⭐⭐")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("### ICLK")
        st.markdown("**Fast & Accurate**")
        st.markdown("Accuracy: ⭐⭐⭐⭐")
        st.markdown("FPS: ⭐⭐⭐⭐")
        st.markdown("Robustness: ⭐⭐⭐")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("### Mean Shift")
        st.markdown("**Histogram-based**")
        st.markdown("Accuracy: ⭐⭐")
        st.markdown("FPS: ⭐⭐⭐")
        st.markdown("Robustness: ⭐⭐")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### MOSSE Tracker Advantages
        - Highest tracking accuracy across datasets
        - Real-time performance (60+ FPS)
        - Adaptive correlation filter handles appearance changes
        - Best for animals with distinctive patterns

        ### When MOSSE Struggles
        - Extreme occlusions
        - Very rapid motion
        - Significant scale changes
        """)

    with col2:
        st.markdown("""
        ### ICLK Tracker Characteristics
        - Close second in IoU performance
        - Efficient due to precomputation
        - Requires textured regions
        - Good for structured environments

        ### Mean Shift Challenges
        - Struggles with grouped animals
        - Background confusion
        - Limited performance in complex scenes
        """)

    st.markdown("---")

    st.subheader("Kalman Filter Impact")

    st.markdown("""
    The Kalman filter provides:
    - **Trajectory Smoothing**: Reduces jitter in bounding box movement
    - **Brief Occlusion Handling**: Maintains tracking during short failures
    - **Improved Accuracy**: Modest improvements when base tracker is good

    **Important Note**: Kalman performance is highly dependent on the quality of the underlying tracker.
    The best results come from Kalman + MOSSE combination.
    """)

    st.markdown("---")

    st.subheader("Future Improvements")

    st.markdown("""
    Based on project findings, recommended next steps include:

    1. **Hybrid Meta-Tracker**: Combine multiple trackers for robustness
    2. **YOLO Integration**: Use deep learning for detection and re-identification
    3. **Failure Detection**: Implement PSR (Peak-to-Sidelobe Ratio) monitoring
    4. **Additional Trackers**: Explore KCF and other correlation filter variants
    5. **Re-identification Strategy**: Handle long-term occlusions and reappearances
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; padding: 2rem;">
    <p><strong>Wildlife Tracking with Classical Computer Vision</strong></p>
    <p>CMPUT 428 Project | University of Alberta</p>
    <p>Implemented by Amogh Panhale</p>
</div>
""", unsafe_allow_html=True)
