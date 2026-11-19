import numpy as np
import pytest

from industrial_vision.detector import Detection
from industrial_vision.tracker import ByteTrackTracker, TrackedDetection


def make_detection():
    return Detection(
        x1=1,
        y1=2,
        x2=30,
        y2=40,
        class_id=0,
        class_name="person",
        confidence=0.91,
    )


def test_tracked_detection_accepts_valid_detection_and_id():
    detection = make_detection()

    tracked = TrackedDetection(detection=detection, tracking_id=7)

    assert tracked.detection is detection
    assert tracked.tracking_id == 7


def test_tracked_detection_accepts_zero_id():
    assert TrackedDetection(make_detection(), 0).tracking_id == 0


def test_tracked_detection_accepts_positive_id():
    assert TrackedDetection(make_detection(), 42).tracking_id == 42


def test_tracked_detection_rejects_negative_id():
    with pytest.raises(ValueError, match="non-negative integer"):
        TrackedDetection(make_detection(), -1)


@pytest.mark.parametrize("tracking_id", [True, False])
def test_tracked_detection_rejects_boolean_id(tracking_id):
    with pytest.raises(TypeError, match="non-negative integer"):
        TrackedDetection(make_detection(), tracking_id)


@pytest.mark.parametrize("tracking_id", [1.5, "1", None, 1 + 0j])
def test_tracked_detection_rejects_invalid_id_types(tracking_id):
    with pytest.raises(TypeError, match="non-negative integer"):
        TrackedDetection(make_detection(), tracking_id)


@pytest.mark.parametrize("detection", [None, object(), np.zeros((2, 2))])
def test_tracked_detection_rejects_invalid_detection(detection):
    with pytest.raises(TypeError, match="Detection instance"):
        TrackedDetection(detection, 1)


def test_tracked_detection_is_immutable():
    tracked = TrackedDetection(make_detection(), 5)

    with pytest.raises(AttributeError):
        tracked.tracking_id = 6

    with pytest.raises(AttributeError):
        tracked.detection = make_detection()


def test_wrapping_detection_does_not_change_original_detection():
    detection = make_detection()
    original_values = (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
        detection.class_id,
        detection.class_name,
        detection.confidence,
    )

    tracked = TrackedDetection(detection, 3)

    assert tracked.detection is detection
    assert (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
        detection.class_id,
        detection.class_name,
        detection.confidence,
    ) == original_values


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
    def __init__(self, xyxy, class_ids, confidences, tracking_ids):
        self.xyxy = FakeTensor(np.asarray(xyxy).reshape((-1, 4)))
        self.cls = FakeTensor(np.asarray(class_ids).reshape((-1,)))
        self.conf = FakeTensor(np.asarray(confidences).reshape((-1,)))
        self.id = (
            None
            if tracking_ids is None
            else FakeTensor(np.asarray(tracking_ids).reshape((-1,)))
        )


class FakeTrackResult:
    def __init__(self, xyxy, class_ids, confidences, tracking_ids, names):
        self.boxes = FakeBoxes(xyxy, class_ids, confidences, tracking_ids)
        self.names = names


class FakeTrackModel:
    def __init__(self, results_by_call):
        self.results_by_call = iter(results_by_call)
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.results_by_call)


def make_track_result(
    xyxy,
    class_ids,
    confidences,
    tracking_ids,
    names=None,
):
    return FakeTrackResult(
        xyxy=xyxy,
        class_ids=class_ids,
        confidences=confidences,
        tracking_ids=tracking_ids,
        names={0: "person", 2: "forklift"} if names is None else names,
    )


def test_byte_track_model_is_loaded_lazily_and_reused_for_multiple_frames():
    empty_result = make_track_result([], [], [], [], names={})
    model = FakeTrackModel([[empty_result], [empty_result]])
    factory_calls = []

    def model_factory(checkpoint):
        factory_calls.append(checkpoint)
        return model

    tracker = ByteTrackTracker("custom.pt", model_factory=model_factory)

    assert factory_calls == []
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    assert tracker.track(frame) == []
    assert tracker.track(frame) == []

    assert factory_calls == ["custom.pt"]
    assert len(model.calls) == 2


def test_byte_track_uses_configured_cpu_tracking_arguments():
    result = make_track_result([], [], [], [], names={})
    model = FakeTrackModel([[result]])
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker = ByteTrackTracker(
        confidence_threshold=0.1,
        model_factory=lambda _: model,
    )

    tracker.track(frame)

    call = model.calls[0]
    assert call["source"] is frame
    assert call["persist"] is True
    assert call["tracker"] == "bytetrack.yaml"
    assert call["conf"] == 0.1
    assert call["device"] == "cpu"
    assert call["verbose"] is False


def test_byte_track_converts_boxes_and_tracking_ids_to_project_models():
    result = make_track_result(
        xyxy=[[1.5, 2.0, 30.0, 40.5], [10, 11, 50, 60]],
        class_ids=[0.0, 2.0],
        confidences=[0.25, 0.82],
        tracking_ids=[8, 21],
    )
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([[result]]))

    tracked = tracker.track(np.zeros((80, 90, 3), dtype=np.uint8))

    assert tracked == [
        TrackedDetection(
            Detection(1.5, 2.0, 30.0, 40.5, 0, "person", 0.25),
            8,
        ),
        TrackedDetection(
            Detection(10.0, 11.0, 50.0, 60.0, 2, "forklift", 0.82),
            21,
        ),
    ]


def test_byte_track_preserves_large_unsigned_integer_tracking_id():
    large_tracking_id = np.uint64(9007199254740993)
    result = make_track_result(
        xyxy=[[1, 2, 30, 40]],
        class_ids=[0],
        confidences=[0.9],
        tracking_ids=[large_tracking_id],
    )
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([[result]]))

    tracked = tracker.track(np.zeros((50, 50, 3), dtype=np.uint8))

    assert tracked[0].tracking_id == 9007199254740993


def test_byte_track_accepts_sequence_class_names():
    result = make_track_result(
        xyxy=[[1, 2, 30, 40]],
        class_ids=[1],
        confidences=[0.9],
        tracking_ids=[4],
        names=["person", "worker"],
    )
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([[result]]))

    tracked = tracker.track(np.zeros((50, 50, 3), dtype=np.uint8))

    assert tracked[0].detection.class_name == "worker"


def test_byte_track_returns_empty_for_no_results_or_missing_tracking_ids():
    missing_ids = make_track_result(
        xyxy=[[1, 2, 30, 40]],
        class_ids=[0],
        confidences=[0.9],
        tracking_ids=None,
    )
    model = FakeTrackModel([[], [missing_ids]])
    tracker = ByteTrackTracker(model_factory=lambda _: model)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    assert tracker.track(frame) == []
    assert tracker.track(frame) == []


def test_byte_track_skips_invalid_ids_without_inventing_replacements():
    result = make_track_result(
        xyxy=[
            [1, 2, 30, 40],
            [2, 3, 31, 41],
            [3, 4, 32, 42],
            [4, 5, 33, 43],
            [5, 6, 34, 44],
        ],
        class_ids=[0, 0, 0, 0, 0],
        confidences=[0.9, 0.8, 0.7, 0.6, 0.5],
        tracking_ids=[12, np.nan, -1, np.inf, 1.5],
    )
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([[result]]))

    tracked = tracker.track(np.zeros((50, 50, 3), dtype=np.uint8))

    assert [item.tracking_id for item in tracked] == [12]


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (None, TypeError),
        (np.zeros((20, 30), dtype=np.uint8), ValueError),
        (np.zeros((20, 30, 4), dtype=np.uint8), ValueError),
        (np.zeros((0, 30, 3), dtype=np.uint8), ValueError),
        (np.zeros((20, 30, 3), dtype=np.float32), TypeError),
    ],
)
def test_byte_track_rejects_invalid_frames(frame, error):
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([]))

    with pytest.raises(error):
        tracker.track(frame)


@pytest.mark.parametrize(
    ("threshold", "error"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.1", TypeError),
    ],
)
def test_byte_track_rejects_invalid_confidence_thresholds(threshold, error):
    with pytest.raises(error):
        ByteTrackTracker(confidence_threshold=threshold)


@pytest.mark.parametrize(
    ("checkpoint", "error"),
    [
        ("", ValueError),
        ("  ", ValueError),
        (None, TypeError),
    ],
)
def test_byte_track_rejects_invalid_checkpoint_configuration(checkpoint, error):
    with pytest.raises(error):
        ByteTrackTracker(checkpoint=checkpoint)


def test_byte_track_rejects_non_callable_model_factory():
    with pytest.raises(TypeError, match="callable"):
        ByteTrackTracker(model_factory=object())


def test_byte_track_rejects_factory_result_without_track_method():
    tracker = ByteTrackTracker(model_factory=lambda _: object())

    with pytest.raises(TypeError, match="track method"):
        tracker.track(np.zeros((20, 30, 3), dtype=np.uint8))


def test_byte_track_does_not_mutate_input_frame_or_returned_detections():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    original_frame = frame.copy()
    result = make_track_result(
        xyxy=[[1, 2, 30, 40]],
        class_ids=[0],
        confidences=[0.9],
        tracking_ids=[5],
    )
    tracker = ByteTrackTracker(model_factory=lambda _: FakeTrackModel([[result]]))

    tracked = tracker.track(frame)

    np.testing.assert_array_equal(frame, original_frame)
    assert tracked[0].detection == Detection(1, 2, 30, 40, 0, "person", 0.9)
    with pytest.raises(AttributeError):
        tracked[0].detection.class_name = "changed"
