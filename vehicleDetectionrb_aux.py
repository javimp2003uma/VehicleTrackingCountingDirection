import argparse
import os
from typing import Dict, Iterable, List, Set

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


ZONE_IN_POLYGONS = [
    np.array([[652, 214], [795, 62], [947, 205], [804, 357]]),
    np.array([[1275, 384], [1418, 232], [1570, 375], [1427, 527]]),
    np.array([[1127, 933], [1271, 781], [1423, 925], [1279, 1077]]),
    np.array([[478, 753], [621, 601], [773, 744], [630, 896]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[622, 555], [766, 403], [614, 260], [470, 412]]),         # red
    np.array([[1052, 164], [1195, 12], [1347, 155], [1204, 307]]),      # green
    np.array([[1302, 754], [1445, 602], [1597, 745], [1454, 897]]),     # yellow
    np.array([[654, 931], [798, 779], [950, 922], [806, 1074]]),        # blue
]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        roboflow_api_key: str,
        model_id: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        height, width, _ = annotated_frame.shape

        # Define the size of the boxes
        box_size = 150  # width and height of the square boxes
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font for text
        font_scale = 1
        color = (255, 255, 255)  # Red color in BGR
        thickness = 2  # Thickness for both text and box lines

        # Coordinates for the four corners: North-West, South-West, North-East, South-East
        corners = {
            'North': (10, 10),
            'East': (width - box_size - 10, 10),
            'South': (width - box_size - 10, height - box_size - 10),
            'West': (10, height - box_size - 10)
        }

        # Draw the boxes and text in each corner
        for direction, (x, y) in corners.items():
            # Draw the rectangle
            cv2.rectangle(annotated_frame, (x, y), (x + box_size, y + box_size), color, -1)
            # Put the direction text in the superior part of the box
            text_size = cv2.getTextSize(direction, font, font_scale, thickness)[0]
            text_x = x + (box_size - text_size[0]) // 2  # Center horizontally
            text_y = y + text_size[1] + 5  # Position near the top, with 5 pixels margin from the top
            cv2.putText(annotated_frame, direction, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

                    _, actualValue = list(corners.items())[zone_out_id]
                    x = actualValue[0]
                    y = actualValue[1]
                    
                    text_size = cv2.getTextSize(str(count), font, font_scale, thickness)[0]
                    #text_x = x + (box_size - text_size[0]) // 2     # Center horizontally
                    text_x = x + 10 + i * (text_size[0] + 10)
                    text_y = (y + box_size) - text_size[1] - 5      # Position near the bottom, with 5 pixels margin from it

                    print(f"printing count: {count} in {text_x}, {text_y}")
                    color = COLORS.colors[zone_in_id]

                    #cv2.putText(annotated_frame, str(count), (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(annotated_frame, str(count), (text_x, text_y), font, font_scale, (color.b, color.g, color.r), thickness)


        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.infer(
            frame, confidence=self.conf_threshold, iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(results)

        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )

        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with Inference and ByteTrack"
    )

    parser.add_argument(
        "--model_id",
        default="skyview-vehicle/4",
        help="Roboflow model ID",
        type=str,
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API KEY",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API KEY is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    args.roboflow_api_key = api_key

    processor = VideoProcessor(
        roboflow_api_key=args.roboflow_api_key,
        model_id=args.model_id,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
