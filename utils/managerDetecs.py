from typing import List, Dict, Tuple
import numpy as np
import supervision as sv

class DetMan:
    def __init__(self) -> None:
        self.tracker_id_to_zone_status: Dict[int, bool] = {}  # Tracker ID to whether the object is inside the zone
        self.count_inside: int = 0  # Count of objects currently inside the zone
        self.previous_positions: Dict[int, Tuple[float, float]] = {}  # Tracker ID to (x, y)
        self.speeds: Dict[int, float] = {}  # Tracker ID to speed

    def update_positions(self, detections: sv.Detections):
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            self.previous_positions[tracker_id] = (x_center, y_center)

    def calculate_speed(self, tracker_id, new_position, frame_rate, scale):
        if tracker_id not in self.previous_positions:
            return 0
        old_position = self.previous_positions[tracker_id]
        distance_pixels = np.sqrt((new_position[0] - old_position[0])**2 + (new_position[1] - old_position[1])**2)
        distance_real = distance_pixels * scale  # Convert pixel distance to real-world distance
        self.speeds[tracker_id] = distance_real * frame_rate
        return self.speeds[tracker_id] 

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zone: List[sv.Detections],
    ) -> sv.Detections:
        
        # detections_all is just Detections object
        # detections_in_zones is a List<Detections>, just 1 element

        # Track objects entering the zone
        for tracker_id in detections_in_zone[0].tracker_id:
            self.tracker_id_to_zone_status[tracker_id] = True
            self.count_inside += 1

        nonInzone = [item for item in detections_all.tracker_id if item not in detections_in_zone[0].tracker_id]
        for i in nonInzone:
            self.tracker_id_to_zone_status[i] = False

        # Track objects leaving the zone
        for tracker_id in detections_all.tracker_id:
            if tracker_id in self.tracker_id_to_zone_status and not self.tracker_id_to_zone_status[tracker_id]:
                self.count_inside -= 1
        
        # Mark objects that are outside the zone
        for tracker_id in detections_all.tracker_id:
            if tracker_id in self.tracker_id_to_zone_status:
                self.tracker_id_to_zone_status[tracker_id] = False

        # Update detections' class IDs based on zone status
        detections_all.class_id = np.vectorize(
            lambda x: 0 if self.tracker_id_to_zone_status.get(x, False) else -1
        )(detections_all.tracker_id)

        return detections_all[detections_all.class_id != -1]
