# Vehicle Tracking, Counting, and Direction Detection

This project is focused on detecting, tracking, counting vehicles, and determining their direction of movement in video footage using YOLO and ByteTrack.



## Project Overview

The goal of this project is to:
- Detect vehicles using the YOLO object detection model.
- Track detected vehicles across frames using the ByteTrack tracker.
- Count the number of vehicles entering and exiting defined zones.
- Identify the direction of the vehicle's movement (upwards or downwards in the frame).

## Features

- **Vehicle Detection**: Uses YOLOv8 for detecting vehicles.
- **Vehicle Tracking**: Uses ByteTrack to track vehicle movements across frames.
- **Vehicle Counting**: Counts vehicles that enter or exit specific zones.
- **Direction Detection**: Determines whether vehicles are moving upwards (into the zone) or downwards (exiting the zone).

## Installation

### Prerequisites
- Python 3.10 or higher
- A virtual environment (optional but recommended)

### Required Libraries

To install the required dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-tracking.git
   cd vehicle-tracking
    ```
2. Create a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
    ```
4. Download neccesary files:
   ```bash
   ./downloadFiles.sh
    ```
5. Run it:
    ```bash
   python3 inference.py --args
    ```
## Configuration

- **Zone Setup**: Define `ZONE_IN_POLYGONS` and `ZONE_OUT_POLYGONS` in `video_processing/utils.py` to specify the areas for tracking vehicle entry and exit.

## Acknowledgements

- YOLOv8 by Ultralytics for vehicle detection.
- ByteTrack for object tracking.
- Supervision Library for annotation and utilities.
