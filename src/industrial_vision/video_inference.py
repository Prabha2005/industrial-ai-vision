"""Frame-by-frame object detection for recorded videos."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from industrial_vision.detector import (
    Detection,
    YoloDetector,
    filter_detections_by_confidence,
    non_max_suppression,
)
from industrial_vision.image_processing import annotate_image
from industrial_vision.tracker import (
    ByteTrackTracker,
    TrackedDetection,
)

Image = NDArray[np.uint8]
FrameSize = tuple[int, int]
OUTPUT_CODECS = {".avi": "XVID", ".mp4": "mp4v"}
FALLBACK_FPS = 30.0


class FrameDetector(Protocol):
    def detect(self, image: Image) -> list[Detection]: ...


class FrameTracker(Protocol):
    def track(self, frame: Image) -> list[TrackedDetection]: ...


class VideoCapture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Image | None]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


class VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, image: Image) -> None: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str], VideoCapture]
WriterFactory = Callable[[str, int, float, FrameSize], VideoWriter]
TrackerFactory = Callable[[float], FrameTracker]


@dataclass(frozen=True, slots=True)
class VideoInferenceSummary:
    frames_processed: int
    raw_detections: int | None
    confidence_filtered_detections: int | None
    nms_detections: int | None
    average_processing_seconds_per_frame: float
    output_path: Path
    tracking_enabled: bool = False
    tracked_detections: int | None = None


def _validate_unit_interval(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return normalized


def _validate_video_paths(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path, str]:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Input path is not a video file: {source}")

    destination = Path(output_path)
    codec_name = OUTPUT_CODECS.get(destination.suffix.lower())
    if codec_name is None:
        raise ValueError("output_path must have an .mp4 or .avi extension")
    if source.resolve() == destination.resolve():
        raise ValueError("input and output video paths must be different")
    if destination.exists() and destination.is_dir():
        raise ValueError(f"Output path is a directory: {destination}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Could not create output directory: {destination.parent}"
        ) from error
    return source, destination, codec_name


def _validate_frame(frame: object, expected_size: FrameSize | None = None) -> Image:
    if not isinstance(frame, np.ndarray):
        raise TypeError("video frames must be NumPy arrays")
    if frame.dtype != np.uint8:
        raise TypeError("video frames must have dtype uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("video frames must be non-empty BGR images")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("video frames must be non-empty BGR images")
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError("video frame dimensions changed during processing")
    return frame


def _annotate_detections(frame: Image, detections: list[Detection]) -> Image:
    annotated = frame
    for detection in detections:
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
    return annotated


def _annotate_tracked_detections(
    frame: Image,
    detections: list[TrackedDetection],
) -> Image:
    annotated = frame
    for tracked_detection in detections:
        detection = tracked_detection.detection
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
            f"{detection.class_name} ID:{tracked_detection.tracking_id}",
            confidence=detection.confidence,
        )
    return annotated


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    detector: FrameDetector | None = None,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
    tracking: bool = False,
    tracking_confidence_threshold: float | None = None,
    tracker_factory: TrackerFactory | None = None,
    capture_factory: CaptureFactory | None = None,
    writer_factory: WriterFactory | None = None,
) -> VideoInferenceSummary:
    """Process a video in detection-only or persistent ByteTrack mode.

    The reported average covers inference, post-processing, annotation, and
    writing, but excludes video decoding and model initialization. Invalid or
    unavailable input FPS falls back to 30 FPS. Output supports MP4 and AVI.

    Detection-only mode is the default and applies project confidence filtering
    and NMS. Tracking mode sends each frame directly to one per-video tracker;
    detection-only thresholds are not accepted in that mode.
    """
    if not isinstance(tracking, bool):
        raise TypeError("tracking must be a boolean")
    if tracking:
        if confidence_threshold is not None or iou_threshold is not None:
            raise ValueError(
                "confidence_threshold and iou_threshold are detection-only "
                "options and cannot be set in tracking mode"
            )
        if tracker_factory is not None and not callable(tracker_factory):
            raise TypeError("tracker_factory must be callable")
        tracking_confidence = _validate_unit_interval(
            0.1
            if tracking_confidence_threshold is None
            else tracking_confidence_threshold,
            "tracking_confidence_threshold",
        )
    else:
        if tracking_confidence_threshold is not None:
            raise ValueError(
                "tracking_confidence_threshold can only be set in tracking mode"
            )
        if tracker_factory is not None:
            raise ValueError("tracker_factory can only be set in tracking mode")
        effective_confidence_threshold = (
            0.70 if confidence_threshold is None else confidence_threshold
        )
        effective_iou_threshold = 0.50 if iou_threshold is None else iou_threshold
        filter_detections_by_confidence((), effective_confidence_threshold)
        non_max_suppression((), effective_iou_threshold)

    source, destination, codec_name = _validate_video_paths(
        input_path,
        output_path,
    )

    if not tracking and detector is None:
        detector = YoloDetector()
    if capture_factory is None:
        capture_factory = cv2.VideoCapture
    if writer_factory is None:
        writer_factory = cv2.VideoWriter

    capture = capture_factory(str(source))
    writer: VideoWriter | None = None
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {source}")

        has_frame, first_frame = capture.read()
        if not has_frame:
            raise OSError(f"Video contains no readable frames: {source}")
        frame = _validate_frame(first_frame)
        height, width = frame.shape[:2]
        frame_size = (width, height)

        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        output_fps = (
            reported_fps
            if math.isfinite(reported_fps) and reported_fps > 0
            else FALLBACK_FPS
        )
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = writer_factory(str(destination), fourcc, output_fps, frame_size)
        if not writer.isOpened():
            raise OSError(f"Could not initialize video writer: {destination}")

        frame_count = 0
        raw_count = 0
        confidence_filtered_count = 0
        nms_count = 0
        tracked_count = 0
        processing_seconds = 0.0
        frame_tracker: FrameTracker | None = None
        if tracking:
            if tracker_factory is None:
                frame_tracker = ByteTrackTracker(
                    confidence_threshold=tracking_confidence,
                )
            else:
                frame_tracker = tracker_factory(tracking_confidence)
            if not callable(getattr(frame_tracker, "track", None)):
                raise TypeError("tracker_factory must return an object with track()")

        while True:
            frame = _validate_frame(frame, frame_size)
            started_at = perf_counter()

            if tracking:
                tracked_detections = frame_tracker.track(frame)
                writer.write(
                    _annotate_tracked_detections(frame.copy(), tracked_detections)
                )
                tracked_count += len(tracked_detections)
            else:
                raw_detections = detector.detect(frame)
                confidence_filtered = filter_detections_by_confidence(
                    raw_detections,
                    effective_confidence_threshold,
                )
                final_detections = non_max_suppression(
                    confidence_filtered,
                    effective_iou_threshold,
                )
                writer.write(_annotate_detections(frame.copy(), final_detections))
                raw_count += len(raw_detections)
                confidence_filtered_count += len(confidence_filtered)
                nms_count += len(final_detections)

            processing_seconds += perf_counter() - started_at
            frame_count += 1

            has_frame, next_frame = capture.read()
            if not has_frame:
                break
            frame = next_frame

        return VideoInferenceSummary(
            frames_processed=frame_count,
            raw_detections=None if tracking else raw_count,
            confidence_filtered_detections=(
                None if tracking else confidence_filtered_count
            ),
            nms_detections=None if tracking else nms_count,
            average_processing_seconds_per_frame=processing_seconds / frame_count,
            output_path=destination,
            tracking_enabled=tracking,
            tracked_detections=tracked_count if tracking else None,
        )
    finally:
        try:
            if writer is not None:
                writer.release()
        finally:
            capture.release()
