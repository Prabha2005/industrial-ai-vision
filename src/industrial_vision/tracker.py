"""Project-owned object tracking results and the Ultralytics ByteTrack adapter."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import math
from numbers import Integral, Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from industrial_vision.detector import Detection


Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection associated with a non-negative tracking identifier."""

    detection: Detection
    tracking_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.detection, Detection):
            raise TypeError("detection must be a Detection instance")
        if isinstance(self.tracking_id, bool) or not isinstance(
            self.tracking_id, Integral
        ):
            raise TypeError("tracking_id must be a non-negative integer")
        if self.tracking_id < 0:
            raise ValueError("tracking_id must be a non-negative integer")
        object.__setattr__(self, "tracking_id", int(self.tracking_id))


class _TrackBoxes(Protocol):
    xyxy: object
    cls: object
    conf: object
    id: object


class _TrackResult(Protocol):
    boxes: _TrackBoxes
    names: Mapping[int, str] | Sequence[str]


class _YoloTrackModel(Protocol):
    def track(
        self,
        source: Image,
        *,
        persist: bool,
        tracker: str,
        conf: float,
        device: str,
        verbose: bool,
    ) -> Sequence[_TrackResult]: ...


ModelFactory = Callable[[str], _YoloTrackModel]


def _validate_confidence_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence_threshold must be a real number")
    try:
        threshold = float(value)
    except OverflowError as error:
        raise ValueError("confidence_threshold must be finite") from error
    if not math.isfinite(threshold):
        raise ValueError("confidence_threshold must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return threshold


def _to_numpy(value: object) -> NDArray[np.generic]:
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    return np.asarray(value)


def _numeric_array(value: object, field_name: str) -> NDArray[np.generic]:
    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"model output boxes.{field_name} must contain real numbers")
    return array


def _class_name(
    names: Mapping[int, str] | Sequence[str],
    class_id: int,
) -> str:
    try:
        if isinstance(names, Mapping):
            name = names[class_id]
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            name = names[class_id]
        else:
            raise TypeError("model class names must be a mapping or sequence")
    except (KeyError, IndexError) as error:
        raise ValueError(f"model has no name for class_id {class_id}") from error
    if not isinstance(name, str):
        raise TypeError(f"model class name for class_id {class_id} must be a string")
    return name


def _tracking_id(value: np.generic) -> int | None:
    try:
        numeric_id = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric_id) or not numeric_id.is_integer():
        return None
    tracking_id = int(numeric_id)
    return tracking_id if tracking_id >= 0 else None


class ByteTrackTracker:
    """Track detections across frames from one video using Ultralytics ByteTrack."""

    def __init__(
        self,
        checkpoint: str = "yolo26n.pt",
        *,
        confidence_threshold: float = 0.1,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not isinstance(checkpoint, str):
            raise TypeError("checkpoint must be a string")
        if not checkpoint.strip():
            raise ValueError("checkpoint must not be empty")
        if model_factory is not None and not callable(model_factory):
            raise TypeError("model_factory must be callable")

        self._checkpoint = checkpoint
        self._confidence_threshold = _validate_confidence_threshold(
            confidence_threshold
        )
        self._model_factory = model_factory
        self._model: _YoloTrackModel | None = None

    def track(self, frame: Image) -> list[TrackedDetection]:
        """Track objects in one BGR uint8 frame without filtering low scores."""
        self._validate_frame(frame)
        results = self._get_model().track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self._confidence_threshold,
            device="cpu",
            verbose=False,
        )
        if results is None:
            return []

        tracked_detections: list[TrackedDetection] = []
        for result in results:
            boxes_result = getattr(result, "boxes", None)
            if boxes_result is None:
                continue
            tracking_ids_value = getattr(boxes_result, "id", None)
            if tracking_ids_value is None:
                continue

            boxes = _numeric_array(boxes_result.xyxy, "xyxy")
            class_ids = _numeric_array(boxes_result.cls, "cls")
            confidences = _numeric_array(boxes_result.conf, "conf")
            tracking_ids = _numeric_array(tracking_ids_value, "id")
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("model output boxes.xyxy must have shape (N, 4)")
            if any(
                values.ndim != 1
                for values in (class_ids, confidences, tracking_ids)
            ):
                raise ValueError("model output classes, scores, and IDs must be 1D")
            if not (
                len(boxes)
                == len(class_ids)
                == len(confidences)
                == len(tracking_ids)
            ):
                raise ValueError("model output boxes, classes, scores, and IDs must align")

            for box, class_id_value, confidence, tracking_id_value in zip(
                boxes,
                class_ids,
                confidences,
                tracking_ids,
                strict=True,
            ):
                tracking_id = _tracking_id(tracking_id_value)
                if tracking_id is None:
                    continue
                numeric_class_id = float(class_id_value)
                if (
                    not math.isfinite(numeric_class_id)
                    or not numeric_class_id.is_integer()
                ):
                    raise ValueError("model output class IDs must be finite integers")
                class_id = int(numeric_class_id)
                x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
                detection = Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=class_id,
                    class_name=_class_name(result.names, class_id),
                    confidence=float(confidence),
                )
                tracked_detections.append(
                    TrackedDetection(
                        detection=detection,
                        tracking_id=tracking_id,
                    )
                )
        return tracked_detections

    def _get_model(self) -> _YoloTrackModel:
        if self._model is not None:
            return self._model
        model_factory = self._model_factory
        if model_factory is None:
            try:
                ultralytics_module = import_module("ultralytics")
            except ImportError as error:
                raise ImportError(
                    "Install the optional Ultralytics dependency to use ByteTrackTracker"
                ) from error
            model_factory = getattr(ultralytics_module, "YOLO", None)
            if not callable(model_factory):
                raise ImportError(
                    "The installed Ultralytics package does not expose YOLO"
                )
        model = model_factory(self._checkpoint)
        if not callable(getattr(model, "track", None)):
            raise TypeError("model_factory must return a model with a track method")
        self._model = model
        return model

    @staticmethod
    def _validate_frame(frame: Image) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.dtype != np.uint8:
            raise TypeError("frame must have dtype uint8")
        if (
            frame.ndim != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
            or frame.shape[2] != 3
        ):
            raise ValueError("frame must be a non-empty BGR image with 3 channels")
