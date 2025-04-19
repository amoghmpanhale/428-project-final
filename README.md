# 428-project-final
This repository contains some of the code for the cmput 428 project for wildlife monitoring. This project implements various wildlife tracking algorithms and compares them to each other on the animal data in the data folder.

## Project Structure
The contents of this repository are organized as is decribed below:
- `src/`: Contains implementations of the Inverse Compositional Lucas Kanade Tracking Algorithm, The mean shift tracking algorithm, and the discrete Kalman Filter.
- `data/`: Contains video data with it's .txt annotations. These are derived from the AnimalTrack Dataset and the Zebra Dataset Cited.
- `outputs/`: Stores some sample outputs of the trackers and filters on different image sequences for a specific animal. 
- `test_*.py`: These scripts in the root repository can be run to evaluate the performance of the tracker they are testing on a .mp4 file with it's ground truth .txt file.
- `utils.py`: Utility functions used across the testing algorithms. References were used for some of the functions inside. These are also cited.
- `README.md`: You are here!

## Installation
To setup this repository on your personal system, follow the steps below.
1. Clone the repository
2. Install the required python packages using `pip install -r requirements.txt`

## Use
To run a specific tracker or implementation of the Kalman filter, run the test scripts. For example:
`python test_mosse.py`
The path to the video and ground truth are located inside of the test file at the top of the script. In order to run the test on a different image sequence, update these static variables and run the script.

## Findings
Preliminary runs of the trackers suggests that there needs to be a more robust implementation. The MOSSE tracker boasts the highest IoU and FPS so far with the Inverse Compositional Lucas Kanade tracker close behind. Initially, it was thought that the mean shift tracker would allow for animal patterns to be easily tracked but animals traveling in groups tend to confuse the tracker to the point of neglible tracking. The performance of the Discrete Kalman Filter is definitely dependent on the performance of the tracker that trains it. Tests with it have shown some minor increases in IoU. 

## Future Work
This project has a lot left to be done. Since the original classical trackers haven't proven to be adequate, the integration of a hybrid meta tracker and a re-identification strategy is necessary. Creator notes on it suggest:
- Implementing a trained YOLO model to detect animals and reidentify with the detected bounding box that is closest to the last traked region
- Implement a means to tell if the tracker is failing or not. Re-identification models would help with this. The Peak-to-Sidelobe ratio of the MOSSE tracker would also suffice. Future work should include a custom implementation of the tracker so as to be able to access the PSR value. The legacy opencv tracker implemented currently does not allow for that.
- More trackers should be implemented. The promise shown by the correlational filter based MOSSE tracker suggests that the KCF tracker could also be used. 

## References
1. David S Bolme, J Ross Beveridge, Bruce A Draper, and Yui Man Lui. Visual object
tracking using adaptive correlation filters. In 2010 IEEE computer society conference on
computer vision and pattern recognition, pages 2544–2550. IEEE, 2010.
2. Simon Baker and Iain Matthews. Lucas-kanade 20 years on: A unifying framework.
International journal of computer vision, 56:221–255, 2004
3. Dorin Comaniciu, Visvanathan Ramesh, and Peter Meer. Real-time tracking of non-
rigid objects using mean shift. In Proceedings IEEE Conference on Computer Vision
and Pattern Recognition. CVPR 2000 (Cat. No. PR00662), volume 2, pages 142–149.
IEEE, 2000.
4. Eric Price, Pranav C Khandelwal, Daniel I Rubenstein, and Aamir Ahmad. A framework
for fast, large-scale, semi-automatic inference of animal behavior from monocular videos.
bioRxiv, pages 2023–07, 2023.
5. Greg Welch, Gary Bishop, et al. An introduction to the kalman filter. 1995.
6. Libo Zhang, Junyuan Gao, Zhen Xiao, and Heng Fan. Animaltrack: A benchmark for
multi-animal tracking in the wild. International Journal of Computer Vision, 131(2):496–
513, 2023.
7. https://www.youtube.com/watch?v=mwn8xhgNpFY&list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr&ab_channel=MATLAB
8. https://github.com/zziz/kalman-filter
9. https://github.com/Blarc/mean-shift-tracking
10. Code from the UAlberta CMPUT 428 Assignments and LAbs was also used

## AI Tool Use Acknowledgement
In the creation of this project I did make use of the language model Claude to aid with detecting typos, and with formatting comments. The code in this repository was written by me though.

