"""Run CPU YOLO detection on a recorded video."""

import argparse
from pathlib import Path

from industrial_vision.detector import (
    YoloDetector,
    filter_detections_by_confidence,
    non_max_suppression,
)
from industrial_vision.video_inference import process_video

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
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "weights" / "yolo26n.pt",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=parse_confidence_threshold,
        default=None,
        help=(
            "Detection-only: keep detections at or above this confidence "
            "(default: 0.70)"
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=parse_iou_threshold,
        default=None,
        help=(
            "Detection-only: suppress same-class detections above this IoU "
            "(default: 0.50)"
        ),
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help=(
            "Enable ByteTrack. Detection-only confidence/IoU options are not "
            "used; ByteTrack receives detections from its own low-confidence "
            "inference path."
        ),
    )
    parser.add_argument(
        "--tracking-confidence-threshold",
        type=parse_confidence_threshold,
        default=None,
        help=(
            "Tracking only: detector input confidence for ByteTrack "
            "(default: 0.10)"
        ),
    )
    args = parser.parse_args()

    if args.tracking:
        if (
            args.confidence_threshold is not None
            or args.iou_threshold is not None
        ):
            parser.error(
                "--confidence-threshold and --iou-threshold are detection-only; "
                "omit them with --tracking"
            )
        detector = None
    else:
        if args.tracking_confidence_threshold is not None:
            parser.error(
                "--tracking-confidence-threshold requires --tracking"
            )
        detector = YoloDetector(checkpoint=str(args.checkpoint))

    summary = process_video(
        args.input_video,
        args.output_video,
        detector=detector,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        tracking=args.tracking,
        tracking_confidence_threshold=args.tracking_confidence_threshold,
    )
    print(f"Total frames processed: {summary.frames_processed}")
    if summary.tracking_enabled:
        print(f"Total tracked detections: {summary.tracked_detections}")
    else:
        print(f"Total raw detections: {summary.raw_detections}")
        print(
            "Total detections after confidence filtering: "
            f"{summary.confidence_filtered_detections}"
        )
        print(f"Total detections after NMS: {summary.nms_detections}")
    print(
        "Average processing time per frame: "
        f"{summary.average_processing_seconds_per_frame:.3f} seconds"
    )
    print(f"Output video: {summary.output_path}")


if __name__ == "__main__":
    main()
