import math

import numpy as np
import pytest

from industrial_vision.detector import Detection, YoloDetector


def make_detection(**overrides):
    values = {
        "x1": 10,
        "y1": 20,
        "x2": 60,
        "y2": 80,
        "class_id": 2,
        "class_name": "worker",
        "confidence": 0.92,
    }
    values.update(overrides)
    return Detection(**values)


def test_detection_stores_typed_box_and_class_data():
    detection = make_detection()

    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (
        10.0,
        20.0,
        60.0,
        80.0,
    )
    assert detection.class_id == 2
    assert detection.class_name == "worker"
    assert detection.confidence == 0.92


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x1", -1),
        ("y1", -1),
        ("x2", math.inf),
        ("y2", math.nan),
        ("x1", "10"),
    ],
)
def test_detection_rejects_invalid_coordinates(field, value):
    with pytest.raises((TypeError, ValueError)):
        make_detection(**{field: value})


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ((10, 20, 10, 80), "positive-area"),
        ((10, 20, 60, 20), "positive-area"),
    ],
)
def test_detection_rejects_zero_area_boxes(box, message):
    x1, y1, x2, y2 = box
    with pytest.raises(ValueError, match=message):
        make_detection(x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.mark.parametrize(
    ("class_id", "error"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_detection_rejects_invalid_class_ids(class_id, error):
    with pytest.raises(error):
        make_detection(class_id=class_id)


@pytest.mark.parametrize(
    ("class_name", "error"),
    [
        ("  ", ValueError),
        (3, TypeError),
    ],
)
def test_detection_rejects_invalid_class_names(class_name, error):
    with pytest.raises(error):
        make_detection(class_name=class_name)


@pytest.mark.parametrize(
    ("confidence", "error"),
    [
        (-0.01, ValueError),
        (1.01, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        ("0.9", TypeError),
        (True, TypeError),
    ],
)
def test_detection_rejects_invalid_confidence(confidence, error):
    with pytest.raises(error):
        make_detection(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_detection_accepts_confidence_boundaries(confidence):
    assert make_detection(confidence=confidence).confidence == confidence


def test_detection_is_immutable():
    detection = make_detection()

    with pytest.raises(AttributeError):
        detection.class_name = "forklift"


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, class_ids, confidences):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(class_ids)
        self.conf = FakeTensor(confidences)


class FakeResult:
    def __init__(self, xyxy, class_ids, confidences, names):
        self.boxes = FakeBoxes(xyxy, class_ids, confidences)
        self.names = names


class FakeYoloModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def test_yolo_detector_converts_model_output_to_detection_objects():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    result = FakeResult(
        xyxy=[[10.5, 20.0, 60.0, 80.5], [2.0, 3.0, 40.0, 50.0]],
        class_ids=[0.0, 2.0],
        confidences=[0.91, 0.75],
        names={0: "person", 2: "car"},
    )
    model = FakeYoloModel([result])
    detector = YoloDetector(model_factory=lambda _: model)

    detections = detector.detect(image)

    assert detections == [
        Detection(10.5, 20.0, 60.0, 80.5, 0, "person", 0.91),
        Detection(2.0, 3.0, 40.0, 50.0, 2, "car", 0.75),
    ]
    assert model.calls[0]["source"] is image
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["verbose"] is False


def test_yolo_detector_returns_empty_list_when_no_objects_are_detected():
    result = FakeResult(
        xyxy=np.empty((0, 4)),
        class_ids=np.empty((0,)),
        confidences=np.empty((0,)),
        names={0: "person"},
    )
    detector = YoloDetector(model_factory=lambda _: FakeYoloModel([result]))

    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (None, TypeError),
        (np.zeros((0, 10, 3), dtype=np.uint8), ValueError),
        (np.zeros((10, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 4), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 3), dtype=np.float32), TypeError),
    ],
)
def test_yolo_detector_rejects_invalid_images(image, error):
    detector = YoloDetector(model_factory=lambda _: FakeYoloModel([]))

    with pytest.raises(error):
        detector.detect(image)


def test_yolo_detector_loads_model_once_and_reuses_it():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    model = FakeYoloModel([])
    factory_calls = []

    def model_factory(checkpoint):
        factory_calls.append(checkpoint)
        return model

    detector = YoloDetector("custom.pt", model_factory=model_factory)

    detector.detect(image)
    detector.detect(image)

    assert factory_calls == ["custom.pt"]
    assert len(model.calls) == 2
