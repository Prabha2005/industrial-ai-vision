import sys

import pytest

from examples import run_video_inference


def test_invalid_input_path_does_not_initialize_yolo(tmp_path, monkeypatch):
    output_path = tmp_path / "output.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_video_inference.py",
            "--input-video",
            str(tmp_path / "missing.mp4"),
            "--output-video",
            str(output_path),
        ],
    )
    model_initializations = []

    def unexpected_detector(*args, **kwargs):
        model_initializations.append((args, kwargs))
        raise AssertionError("YOLO should not initialize for an invalid input")

    monkeypatch.setattr(run_video_inference, "YoloDetector", unexpected_detector)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_video_inference.main()

    assert model_initializations == []


def test_invalid_output_path_does_not_initialize_yolo(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_video_inference.py",
            "--input-video",
            str(input_path),
            "--output-video",
            str(tmp_path / "output.mov"),
        ],
    )
    model_initializations = []

    def unexpected_detector(*args, **kwargs):
        model_initializations.append((args, kwargs))
        raise AssertionError("YOLO should not initialize for an invalid output")

    monkeypatch.setattr(run_video_inference, "YoloDetector", unexpected_detector)

    with pytest.raises(ValueError, match="mp4 or .avi"):
        run_video_inference.main()

    assert model_initializations == []
