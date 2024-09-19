import cv2
from inference.models.utils import get_roboflow_model
from PIL import Image, ImageFilter
import numpy as np
import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
box_annotator = sv.BoxAnnotator(color=COLORS)
label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)

def annotate_frame(frame: np.ndarray, detections: sv.Detections) -> np.ndarray:

    print(f"annotate_frame funcion has been called with {frame} and {detections}")

    annotated_frame = frame.copy()

    labels = detections.class_id.astype(str).tolist()
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections, labels
    )

    return annotated_frame


# Cargar la imagen usando PIL
image1 = Image.open("/home/javimp2003/VehicleTrackingCountingDirection/random_frame.jpg")
#image2 = image1.filter(ImageFilter.GaussianBlur(1))

# Convertir la imagen a un array numpy (ndarray)
image_array1 = np.array(image1)
#image_array2 = np.array(image2)

model = get_roboflow_model(model_id="skyview-vehicle/4", api_key="??") # complete with Roboflow API key

results1 = model.infer(image_array1, confidence=0.3, iou_threshold=0.5)[0]
#results2 = model.infer(image_array2, confidence=0.3, iou_threshold=0.5)[0]

detections1 = sv.Detections.from_inference(results1)
#detections2 = sv.Detections.from_inference(results2)

annotated_frame = annotate_frame(image_array1, detections1)

cv2.imwrite("processedFrame.png", annotated_frame)