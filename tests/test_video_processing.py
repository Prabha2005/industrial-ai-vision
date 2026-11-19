import numpy as np
import pytest

from industrial_vision import image_processing


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def test_extract_video_frames_saves_every_nth_frame(tmp_path, monkeypatch):
    capture = FakeCapture([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)])
    monkeypatch.setattr(image_processing.cv2, "VideoCapture", lambda _: capture)

    saved = image_processing.extract_video_frames("clip.mp4", tmp_path, every_n_frames=2)

    assert [path.name for path in saved] == [
        "frame_000000.jpg",
        "frame_000002.jpg",
        "frame_000004.jpg",
    ]
    assert all(path.exists() for path in saved)
    assert capture.released


def test_extract_video_frames_reports_unreadable_video(tmp_path, monkeypatch):
    capture = FakeCapture([], opened=False)
    monkeypatch.setattr(image_processing.cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(OSError, match="Could not open video"):
        image_processing.extract_video_frames("missing.mp4", tmp_path)
    assert capture.released


def test_extract_video_frames_rejects_nonpositive_interval(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        image_processing.extract_video_frames("clip.mp4", tmp_path, every_n_frames=0)
