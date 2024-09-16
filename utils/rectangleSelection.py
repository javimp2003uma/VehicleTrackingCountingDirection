import cv2
import numpy as np
import random

# Global variables to store the points
points = []

# Mouse callback function to store the points
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_resized, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Random Frame', frame_resized)
        if len(points) == 4:
            cv2.destroyWindow('Random Frame')

# Load the video
video_path = '/home/javimp2003/VehicleTrackingCountingDirection/data/vehiclesTraffic1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Pick a random frame index
random_frame_idx = random.randint(0, total_frames - 1)

# Set the video position to the random frame
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)

# Read the frame
ret, frame = cap.read()

# Release the video capture object
cap.release()

# Resize the frame
resize_scale = 0.5  # Change this value to adjust the size
frame_resized = cv2.resize(frame, (int(frame.shape[1] * resize_scale), int(frame.shape[0] * resize_scale)))

# Show the resized frame and set up mouse callback
cv2.imshow('Random Frame', frame_resized)
cv2.setMouseCallback('Random Frame', click_event)

# Wait until 4 points are selected
while len(points) < 4:
    cv2.waitKey(1)

# Print the points
print('Selected points:', points)

# Destroy all windows
cv2.destroyAllWindows()