"""Run CPU YOLO detection on one image and save an annotated copy."""

import argparse
from pathlib import Path

import cv2

from industrial_vision.detector import (
    YoloDetector,
    filter_detections_by_confidence,
    non_max_suppression,
)
from industrial_vision.image_processing import annotate_image, load_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_confidence_threshold(value: str) -> float:
    try:
        threshold = float(value)
        filter_detections_by_confidence((), threshold)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return threshold


def parse_iou_threshold(value: str) -> float:
    try:
        threshold = float(value)
        non_max_suppression((), threshold)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "data" / "samples" / "bus.jpg",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "weights" / "yolo26n.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detect" / "bus_detected.jpg",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=parse_confidence_threshold,
        default=0.70,
        help="Keep detections with confidence >= this value (0.0-1.0)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=parse_iou_threshold,
        default=0.5,
        help="Suppress same-class detections with IoU greater than this value",
    )
    args = parser.parse_args()

    image = load_image(args.image)
    detector = YoloDetector(checkpoint=str(args.checkpoint))
    raw_detections = detector.detect(image)
    confidence_filtered_detections = filter_detections_by_confidence(
        raw_detections,
        args.confidence_threshold,
    )
    final_detections = non_max_suppression(
        confidence_filtered_detections,
        args.iou_threshold,
    )

    print(f"Raw detections: {len(raw_detections)}")
    print(
        f"After confidence filtering (>= {args.confidence_threshold:.2f}): "
        f"{len(confidence_filtered_detections)}"
    )
    print(
        f"After NMS (suppress same-class IoU > {args.iou_threshold:.2f}): "
        f"{len(final_detections)}"
    )
    if not final_detections:
        print("No detections remain after confidence filtering and NMS.")
    for detection in final_detections:
        print(
            f"{detection.class_name} (class_id={detection.class_id}, "
            f"confidence={detection.confidence:.3f}, "
            f"box=({detection.x1:.1f}, {detection.y1:.1f}, "
            f"{detection.x2:.1f}, {detection.y2:.1f}))"
        )

    annotated = image.copy()
    for detection in final_detections:
        box = tuple(
            round(coordinate)
            for coordinate in (
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            )
        )
        annotated = annotate_image(
            annotated,
            box,
            detection.class_name,
            confidence=detection.confidence,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"Could not write annotated image: {args.output}")
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
