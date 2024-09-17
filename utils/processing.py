from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from utils.managerDetecs import DetMan

POLYGONS = [
    np.array([[2038, 444], [2775, 1108], [2062, 1963], [1332, 1211]])
]
COLORS = sv.ColorPalette.DEFAULT

class Processor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        workingDirectory: str,
        confidence_threshold: float = 0.3
    ) -> None:
        self.model = YOLO(source_weights_path)
        #self.model.to('cuda')
        self.source_video_path = source_video_path
        self.conf_threshold = confidence_threshold
        self.workingDirectory = workingDirectory
        self.countFrames = 0
        self.framesSpeed = 10

        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        pol = sv.PolygonZone(
            polygon=POLYGONS[0],
            frame_resolution_wh=self.video_info.resolution_wh,
            triggering_anchors=sv.Position.CENTER
        )
        self.zones_in = [pol]

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetMan()


    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.countFrames += 1
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        # this add additional id for the detections
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []

        for _, zIn in enumerate(self.zones_in):
            # here I can print the detections (whole ones) and detections_in_zone to see if its working
            detections_in_zone = detections[zIn.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)

        # detections is just a Detections object
        # detections_in_zones is a List<Detections>, but just 1 element
        
        # We call the detection manager with all detections and detections in rectangle (Detections and List<Detections>)
        detections = self.detections_manager.update(
            detections, detections_in_zones
        )
        annotated_frame = self.annotate_frame(frame, detections)
        if self.countFrames % self.framesSpeed == 1:
            self.detections_manager.update_positions(detections)
        return annotated_frame
        

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        # Initialize the labels list
        labels = []
        # Generate the labels list using a list comprehension
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]


        # Continue with annotation using the labels
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for i, zone_in in enumerate(self.zones_in):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )

        # Draw count of vehicles inside the rectangle
        count = self.detections_manager.count_inside
        zone_center = sv.get_polygon_center(polygon=self.zones_in[0].polygon)
        text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40)
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Count: {count}",
            text_anchor=text_anchor,
            background_color=COLORS.colors[0],
        )

        return annotated_frame


    def process_video(self, workingDirectory):
        """
        Process the video to detect and count vehicles, and determine their directions.

        Parameters:
        -----------
        weights_path : str
            Path to the model weights file.
        video_path : str
            Path to the source video file.
        confidence_th : float
            Confidence threshold for the model to filter weak detections.
        """
        
        frameGenerator = sv.get_video_frames_generator(self.source_video_path)

        output_video_path = f"{workingDirectory}/output_video.mp4"
        output_video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        with sv.VideoSink(output_video_path, output_video_info) as sink:
            for frame in tqdm(frameGenerator, total=output_video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                sink.write_frame(annotated_frame)