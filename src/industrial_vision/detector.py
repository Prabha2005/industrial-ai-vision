"""Project-owned object detection result types."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import math
from numbers import Integral, Real
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object, with an absolute-pixel ``xyxy`` box."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    def __post_init__(self) -> None:
        for field_name in ("x1", "y1", "x2", "y2", "confidence"):
            normalized = _finite_float(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)

        if min(self.x1, self.y1, self.x2, self.y2) < 0:
            raise ValueError("box coordinates must be non-negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box coordinates must form a positive-area xyxy box")

        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral):
            raise TypeError("class_id must be a non-negative integer")
        if self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        object.__setattr__(self, "class_id", int(self.class_id))

        if not isinstance(self.class_name, str):
            raise TypeError("class_name must be a string")
        if not self.class_name.strip():
            raise ValueError("class_name must not be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def filter_detections_by_confidence(
    detections: Iterable[Detection],
    threshold: float = 0.70,
) -> list[Detection]:
    """Return detections meeting a validated minimum confidence threshold."""
    normalized_threshold = _finite_float(threshold, "threshold")
    if not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    filtered: list[Detection] = []
    for detection in detections:
        if not isinstance(detection, Detection):
            raise TypeError("detections must contain Detection objects")
        if detection.confidence >= normalized_threshold:
            filtered.append(detection)
    return filtered


def intersection_over_union(first: Detection, second: Detection) -> float:
    """Calculate the IoU of two project-owned xyxy detections."""
    if not isinstance(first, Detection) or not isinstance(second, Detection):
        raise TypeError("IoU inputs must be Detection objects")

    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area


def non_max_suppression(
    detections: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Keep high-confidence detections while suppressing same-class overlaps."""
    normalized_threshold = _finite_float(iou_threshold, "iou_threshold")
    if not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    candidates = list(detections)
    if any(not isinstance(detection, Detection) for detection in candidates):
        raise TypeError("detections must contain Detection objects")

    candidates.sort(key=lambda detection: detection.confidence, reverse=True)
    retained: list[Detection] = []
    while candidates:
        selected = candidates[0]
        retained.append(selected)
        candidates = [
            candidate
            for candidate in candidates[1:]
            if candidate.class_id != selected.class_id
            or intersection_over_union(selected, candidate) <= normalized_threshold
        ]
    return retained


class _Boxes(Protocol):
    xyxy: object
    cls: object
    conf: object


class _ModelResult(Protocol):
    boxes: _Boxes
    names: Mapping[int, str] | Sequence[str]


class _YoloModel(Protocol):
    def predict(
        self,
        source: NDArray[np.uint8],
        *,
        device: str,
        verbose: bool,
    ) -> Sequence[_ModelResult]: ...


ModelFactory = Callable[[str], _YoloModel]


def _to_numpy(value: object) -> NDArray[np.generic]:
    """Convert NumPy arrays or tensor-like model values to a NumPy array."""
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
        raise TypeError(f"model output {field_name} must contain real numbers")
    return array


def _class_name(names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
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


class YoloDetector:
    """Run Ultralytics YOLO on CPU and return project-owned detections."""

    def __init__(
        self,
        checkpoint: str = "yolo26n.pt",
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not isinstance(checkpoint, str):
            raise TypeError("checkpoint must be a string")
        if not checkpoint.strip():
            raise ValueError("checkpoint must not be empty")

        if model_factory is None:
            try:
                ultralytics_module = import_module("ultralytics")
            except ImportError as error:
                raise ImportError(
                    "Install the optional Ultralytics dependency to use YoloDetector"
                ) from error
            model_factory = getattr(ultralytics_module, "YOLO", None)
            if not callable(model_factory):
                raise ImportError(
                    "The installed Ultralytics package does not expose YOLO"
                )

        self._model = model_factory(checkpoint)

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Detect objects in one BGR uint8 image using CPU inference."""
        self._validate_image(image)
        results = self._model.predict(source=image, device="cpu", verbose=False)
        detections: list[Detection] = []

        for result in results:
            boxes = _numeric_array(result.boxes.xyxy, "boxes.xyxy")
            class_ids = _numeric_array(result.boxes.cls, "boxes.cls")
            confidences = _numeric_array(result.boxes.conf, "boxes.conf")

            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("model output boxes.xyxy must have shape (N, 4)")
            if class_ids.ndim != 1 or confidences.ndim != 1:
                raise ValueError("model output class IDs and confidences must be 1D")
            if not (len(boxes) == len(class_ids) == len(confidences)):
                raise ValueError("model output boxes, classes, and scores must align")

            for box, class_id_value, confidence in zip(
                boxes, class_ids, confidences, strict=True
            ):
                numeric_class_id = float(class_id_value)
                if (
                    not math.isfinite(numeric_class_id)
                    or not numeric_class_id.is_integer()
                ):
                    raise ValueError("model output class IDs must be finite integers")
                class_id = int(numeric_class_id)
                x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        class_id=class_id,
                        class_name=_class_name(result.names, class_id),
                        confidence=float(confidence),
                    )
                )

        return detections

    @staticmethod
    def _validate_image(image: NDArray[np.uint8]) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.dtype != np.uint8:
            raise TypeError("image must have dtype uint8")
        if (
            image.ndim != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
            or image.shape[2] != 3
        ):
            raise ValueError("image must be a non-empty BGR image with 3 channels")
