# 3D Facial Reconstruction Pipeline: both calibrated and uncalibrated setup

A  pipeline for 3D face reconstruction from multi-view images using:
- MediaPipe for facial landmark detection
- SIFT features for robust matching
- Structure-from-Motion (SfM) principles
- Poisson surface reconstruction

## Features
 Multi-view 3D reconstruction  
 Facial landmark detection with MediaPipe   combined with SIFT feature 
 Camera pose estimation (5-point algorithm)  
 Texture mapping with color enhancement  
 Comprehensive error analysis  

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/luel-abraha/3D-reconstruction-project.git
   cd 3D-reconstruction-project
2 install dependencies
   pip install -r requirements.txt
3 usage:
   Prepare input:
Place images in /samples/samples directory

4 run reconstruction:
python calibrated.py 

python uncalibrated.py
