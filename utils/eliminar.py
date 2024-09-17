import cv2
import random

points = []
def click_event(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        points.append((x, y))

        if len(points) == 4:
            print("Four points selected:", points)
            cv2.destroyAllWindows()

def select_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_index = random.randint(0, frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, img = cap.read()
    cap.release()

    return img

video_path = '/home/javimp2003/VehicleTrackingCountingDirection/data/vehiclesTraffic1.mp4'  # Replace with the actual path to your video
img = select_random_frame(video_path)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)