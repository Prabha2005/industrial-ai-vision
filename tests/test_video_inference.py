import cv2
import numpy as np
import pytest

from industrial_vision.detector import Detection
from industrial_vision.tracker import TrackedDetection
from industrial_vision import video_inference
from industrial_vision.video_inference import process_video


class FakeCapture:
    def __init__(self, frames, *, opened=True, fps=25.0):
        self.frames = iter(frames)
        self.opened = opened
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            return True, next(self.frames).copy()
        except StopIteration:
            return False, None

    def get(self, property_id):
        assert property_id == cv2.CAP_PROP_FPS
        return self.fps

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, *, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FixedWriterFactory:
    def __init__(self, writer):
        self.writer = writer
        self.arguments = None

    def __call__(self, path, fourcc, fps, frame_size):
        self.arguments = (path, fourcc, fps, frame_size)
        return self.writer


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return self.detections


class FakeTracker:
    def __init__(self, detections_by_frame):
        self.detections_by_frame = iter(detections_by_frame)
        self.frames = []

    def track(self, frame):
        self.frames.append(frame)
        return next(self.detections_by_frame)


def make_detection(
    x1=2,
    y1=2,
    x2=12,
    y2=12,
    *,
    class_id=0,
    confidence=0.9,
):
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=class_id,
        class_name=f"class-{class_id}",
        confidence=confidence,
    )


def create_input_file(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake video")
    return source


def make_tracked_detection(*, confidence=0.9, tracking_id=3):
    return TrackedDetection(
        detection=make_detection(confidence=confidence),
        tracking_id=tracking_id,
    )


def test_process_video_filters_suppresses_and_writes_each_frame(tmp_path):
    source = create_input_file(tmp_path)
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    detections = [
        make_detection(confidence=0.95),
        make_detection(x1=2.5, y1=2.5, x2=12.5, y2=12.5, confidence=0.90),
        make_detection(class_id=1, confidence=0.85),
        make_detection(class_id=2, confidence=0.60),
    ]
    capture = FakeCapture(frames)
    writer = FakeWriter()
    detector = FakeDetector(detections)
    writer_arguments = []

    def writer_factory(*arguments):
        writer_arguments.append(arguments)
        return writer

    result = process_video(
        source,
        tmp_path / "output.mp4",
        detector=detector,
        capture_factory=lambda _: capture,
        writer_factory=writer_factory,
    )

    assert result.frames_processed == 2
    assert result.raw_detections == 8
    assert result.confidence_filtered_detections == 6
    assert result.nms_detections == 4
    assert result.average_processing_seconds_per_frame >= 0.0
    assert result.output_path == tmp_path / "output.mp4"
    assert len(detector.frames) == 2
    assert len(writer.frames) == 2
    assert all(frame.shape == (20, 30, 3) for frame in writer.frames)
    assert all(np.count_nonzero(frame) > 0 for frame in writer.frames)
    assert writer_arguments[0][2:] == (25.0, (30, 20))
    assert capture.released
    assert writer.released


def test_process_video_rejects_missing_input_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        process_video(tmp_path / "missing.mp4", tmp_path / "output.mp4")


def test_process_video_rejects_non_file_input(tmp_path):
    with pytest.raises(ValueError, match="not a video file"):
        process_video(tmp_path, tmp_path / "output.mp4")


def test_process_video_rejects_unsupported_output_extension(tmp_path):
    source = create_input_file(tmp_path)

    with pytest.raises(ValueError, match="mp4 or .avi"):
        process_video(source, tmp_path / "output.mov")


def test_process_video_rejects_output_directory_conflict(tmp_path):
    source = create_input_file(tmp_path)
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file", encoding="utf-8")

    with pytest.raises(OSError, match="Could not create output directory"):
        process_video(source, parent_file / "output.mp4")


def test_process_video_rejects_same_input_and_output_path(tmp_path):
    source = create_input_file(tmp_path)

    with pytest.raises(ValueError, match="must be different"):
        process_video(source, source)


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("confidence_threshold", -0.1, ValueError),
        ("confidence_threshold", True, TypeError),
        ("iou_threshold", 1.1, ValueError),
        ("iou_threshold", float("nan"), ValueError),
        ("iou_threshold", False, TypeError),
    ],
)
def test_process_video_rejects_invalid_thresholds(
    tmp_path,
    argument,
    value,
    error,
):
    with pytest.raises(error):
        process_video(
            tmp_path / "missing.mp4",
            tmp_path / "output.mp4",
            **{argument: value},
        )


def test_process_video_rejects_unopenable_video_and_releases_capture(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([], opened=False)

    with pytest.raises(OSError, match="Could not open video"):
        process_video(
            source,
            tmp_path / "output.mp4",
            detector=FakeDetector([]),
            capture_factory=lambda _: capture,
        )

    assert capture.released


def test_process_video_rejects_video_without_readable_frames(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([])

    with pytest.raises(OSError, match="no readable frames"):
        process_video(
            source,
            tmp_path / "output.mp4",
            detector=FakeDetector([]),
            capture_factory=lambda _: capture,
        )

    assert capture.released


def test_process_video_uses_fallback_fps_when_source_fps_is_invalid(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture(
        [np.zeros((20, 30, 3), dtype=np.uint8)],
        fps=float("nan"),
    )
    writer = FakeWriter()
    writer_factory = FixedWriterFactory(writer)

    process_video(
        source,
        tmp_path / "output.mp4",
        detector=FakeDetector([]),
        capture_factory=lambda _: capture,
        writer_factory=writer_factory,
    )

    assert writer_factory.arguments is not None
    assert writer_factory.arguments[2] == 30.0
    assert capture.released
    assert writer.released


def test_process_video_releases_writer_and_capture_on_writer_failure(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([np.zeros((20, 30, 3), dtype=np.uint8)])
    writer = FakeWriter(opened=False)
    writer_factory = FixedWriterFactory(writer)

    with pytest.raises(OSError, match="Could not initialize video writer"):
        process_video(
            source,
            tmp_path / "output.mp4",
            detector=FakeDetector([]),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
        )

    assert capture.released
    assert writer.released
    assert writer_factory.arguments is not None


def test_process_video_releases_resources_when_detector_raises(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([np.zeros((20, 30, 3), dtype=np.uint8)])
    writer = FakeWriter()
    writer_factory = FixedWriterFactory(writer)

    class FailingDetector:
        def detect(self, frame):
            assert frame.shape == (20, 30, 3)
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        process_video(
            source,
            tmp_path / "output.mp4",
            detector=FailingDetector(),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
        )

    assert capture.released
    assert writer.released
    assert writer_factory.arguments is not None


def test_process_video_rejects_frame_size_changes(tmp_path):
    source = create_input_file(tmp_path)
    frames = [
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.zeros((18, 30, 3), dtype=np.uint8),
    ]
    capture = FakeCapture(frames)
    writer = FakeWriter()
    writer_factory = FixedWriterFactory(writer)

    with pytest.raises(ValueError, match="dimensions changed"):
        process_video(
            source,
            tmp_path / "output.mp4",
            detector=FakeDetector([]),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
        )

    assert capture.released
    assert writer.released
    assert writer_factory.arguments is not None


def test_tracking_mode_uses_tracker_without_detection_filter_or_nms(
    tmp_path,
    monkeypatch,
):
    source = create_input_file(tmp_path)
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    tracked_detection = make_tracked_detection(confidence=0.2, tracking_id=17)
    tracker = FakeTracker([[tracked_detection], [tracked_detection]])
    capture = FakeCapture(frames)
    writer = FakeWriter()
    detector = FakeDetector([])
    factory_calls = []

    def tracker_factory(confidence_threshold):
        factory_calls.append(confidence_threshold)
        return tracker

    def unexpected_detection_filter(*args, **kwargs):
        raise AssertionError("detection-only confidence filter ran in tracking mode")

    def unexpected_nms(*args, **kwargs):
        raise AssertionError("detection-only NMS ran in tracking mode")

    monkeypatch.setattr(
        video_inference,
        "filter_detections_by_confidence",
        unexpected_detection_filter,
    )
    monkeypatch.setattr(
        video_inference,
        "non_max_suppression",
        unexpected_nms,
    )
    result = process_video(
        source,
        tmp_path / "tracked.mp4",
        detector=detector,
        tracking=True,
        capture_factory=lambda _: capture,
        writer_factory=lambda *args: writer,
        tracker_factory=tracker_factory,
    )

    assert result.tracking_enabled
    assert result.frames_processed == 2
    assert result.tracked_detections == 2
    assert result.raw_detections is None
    assert result.confidence_filtered_detections is None
    assert result.nms_detections is None
    assert factory_calls == [0.1]
    assert len(tracker.frames) == 2
    assert detector.frames == []
    assert len(writer.frames) == 2
    assert capture.released
    assert writer.released


def test_tracking_mode_annotates_class_and_tracking_id(
    tmp_path,
    monkeypatch,
):
    source = create_input_file(tmp_path)
    tracked = make_tracked_detection(confidence=0.32, tracking_id=24)
    tracker = FakeTracker([[tracked]])
    capture = FakeCapture([np.zeros((20, 30, 3), dtype=np.uint8)])
    writer = FakeWriter()
    annotation_calls = []

    def record_annotation(frame, box, label, confidence=None):
        annotation_calls.append((box, label, confidence))
        return frame

    monkeypatch.setattr(video_inference, "annotate_image", record_annotation)

    process_video(
        source,
        tmp_path / "tracked.mp4",
        tracking=True,
        capture_factory=lambda _: capture,
        writer_factory=lambda *args: writer,
        tracker_factory=lambda _: tracker,
    )

    assert annotation_calls == [((2, 2, 12, 12), "class-0 ID:24", 0.32)]


def test_tracking_mode_handles_empty_results_without_annotation(tmp_path, monkeypatch):
    source = create_input_file(tmp_path)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker = FakeTracker([[]])
    capture = FakeCapture([frame])
    writer = FakeWriter()
    annotation_calls = []
    monkeypatch.setattr(
        video_inference,
        "annotate_image",
        lambda *args, **kwargs: annotation_calls.append(args) or args[0],
    )

    result = process_video(
        source,
        tmp_path / "tracked.mp4",
        tracking=True,
        capture_factory=lambda _: capture,
        writer_factory=lambda *args: writer,
        tracker_factory=lambda _: tracker,
    )

    assert result.tracked_detections == 0
    assert annotation_calls == []
    np.testing.assert_array_equal(writer.frames[0], frame)


def test_tracking_mode_creates_one_fresh_tracker_per_video_session(tmp_path):
    first_source = tmp_path / "first.mp4"
    second_source = tmp_path / "second.mp4"
    first_source.write_bytes(b"video")
    second_source.write_bytes(b"video")
    created_trackers = []

    def tracker_factory(_):
        tracker = FakeTracker([[]])
        created_trackers.append(tracker)
        return tracker

    for source in (first_source, second_source):
        process_video(
            source,
            tmp_path / f"{source.stem}_tracked.mp4",
            tracking=True,
            capture_factory=lambda _: FakeCapture(
                [np.zeros((20, 30, 3), dtype=np.uint8)]
            ),
            writer_factory=lambda *args: FakeWriter(),
            tracker_factory=tracker_factory,
        )

    assert len(created_trackers) == 2
    assert created_trackers[0] is not created_trackers[1]
    assert len(created_trackers[0].frames) == 1
    assert len(created_trackers[1].frames) == 1


def test_tracking_mode_releases_resources_when_tracker_raises(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([np.zeros((20, 30, 3), dtype=np.uint8)])
    writer = FakeWriter()

    class FailingTracker:
        def track(self, frame):
            raise RuntimeError("tracking failed")

    with pytest.raises(RuntimeError, match="tracking failed"):
        process_video(
            source,
            tmp_path / "tracked.mp4",
            tracking=True,
            capture_factory=lambda _: capture,
            writer_factory=lambda *args: writer,
            tracker_factory=lambda _: FailingTracker(),
        )

    assert capture.released
    assert writer.released


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tracking": 1}, TypeError),
        ({"tracking": True, "confidence_threshold": 0.7}, ValueError),
        ({"tracking": True, "iou_threshold": 0.5}, ValueError),
        (
            {"tracking": True, "tracking_confidence_threshold": 1.1},
            ValueError,
        ),
        (
            {"tracking": False, "tracking_confidence_threshold": 0.1},
            ValueError,
        ),
        ({"tracking": False, "tracker_factory": lambda _: None}, ValueError),
        ({"tracking": True, "tracker_factory": object()}, TypeError),
    ],
)
def test_process_video_rejects_invalid_mode_configuration(
    tmp_path,
    kwargs,
    error,
):
    source = create_input_file(tmp_path)

    with pytest.raises(error):
        process_video(source, tmp_path / "output.mp4", **kwargs)


def test_tracking_mode_rejects_invalid_tracker_factory_result(tmp_path):
    source = create_input_file(tmp_path)
    capture = FakeCapture([np.zeros((20, 30, 3), dtype=np.uint8)])
    writer = FakeWriter()

    with pytest.raises(TypeError, match="track"):
        process_video(
            source,
            tmp_path / "tracked.mp4",
            tracking=True,
            capture_factory=lambda _: capture,
            writer_factory=lambda *args: writer,
            tracker_factory=lambda _: object(),
        )

    assert capture.released
    assert writer.released
