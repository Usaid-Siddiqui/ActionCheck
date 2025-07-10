# Cricket Bowling Action Detection

This project detects illegal cricket bowling actions by analyzing elbow extension angles from video footage. It uses **MediaPipe** for pose estimation and **OpenCV** for video processing to track arm joints and calculate elbow angles.

## Features

- Supports tracking of left or right bowling arm  
- Handles bowling directions: left-to-right or right-to-left  
- Processes local MP4 video files  
- Live visualization of elbow angles with matplotlib  
- Skeleton and joint landmarks overlaid on video frames  
- Final legality assessment based on maximum elbow extension during bowling  

## Setup

1. Clone the repository.  
2. Install dependencies using the provided requirements file:  
```bash
pip install -r requirements.txt
```
3. Place your bowling video file in the project folder, named as <lastname>.mp4

## Usage
Run the main script and follow the prompts:
```bash
python analyzeVid.py
```

You will be asked to enter:
- The bowling arm to track (left or right)
- The bowler’s last name (which should match the video filename)
- The bowling direction (ltr for left-to-right, or rtl for right-to-left)
- During processing, a window will display the video with pose landmarks and a live elbow angle plot. Press q to quit the video.

## How It Works

- Starts recording elbow angles when the elbow rises above the shoulder and the wrist is above the elbow, and the arm moves forward in the specified direction.
- Stops recording once the elbow angle drops below 75°.
- Analyzes the last significant spike in elbow angles to assess legality:
    - Extension (max angle - min angle) greater than 15° means illegal bowling action.
    - Otherwise, it’s legal.

## License
This project is licensed under the MIT License.

