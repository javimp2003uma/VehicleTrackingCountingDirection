import argparse
from utils.processing import Processor
import os

cd = os.getcwd()

def main():
    parser = argparse.ArgumentParser(
        description="Counting vehicles and detecting their directions"
    )

    parser.add_argument(
        "--weights_path",
        required=True,
        help="Weights file",
        type=str,
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Source video file",
        type=str,
    )
    parser.add_argument(
        "--confidence_th",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    args = parser.parse_args()
    processingObject = Processor(
        source_weights_path = args.weights_path,
        source_video_path = args.video_path,
        confidence_threshold = args.confidence_th,
        workingDirectory = cd
    )
    processingObject.process_video()

if __name__ == "__main__":
    main()