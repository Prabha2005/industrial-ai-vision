import math

import pytest

from industrial_vision.detector import Detection, filter_detections_by_confidence


def make_detections():
    return [
        Detection(0, 0, 10, 10, 0, "person", 0.0),
        Detection(10, 10, 20, 20, 1, "bus", 0.70),
        Detection(20, 20, 30, 30, 2, "car", 0.90),
    ]


def test_threshold_zero_retains_all_detections():
    detections = make_detections()

    assert filter_detections_by_confidence(detections, 0.0) == detections


def test_threshold_point_seventy_retains_qualifying_detections():
    detections = make_detections()

    assert filter_detections_by_confidence(detections, 0.70) == detections[1:]


def test_detection_equal_to_threshold_is_retained():
    detection = make_detections()[1]

    assert filter_detections_by_confidence([detection], 0.70) == [detection]


def test_filter_returns_no_detections_when_none_qualify():
    assert filter_detections_by_confidence(make_detections(), 0.95) == []


def test_filter_accepts_empty_input():
    assert filter_detections_by_confidence([], 0.70) == []


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf])
def test_filter_rejects_threshold_outside_valid_range(threshold):
    with pytest.raises(ValueError):
        filter_detections_by_confidence(make_detections(), threshold)


@pytest.mark.parametrize("threshold", [True, False, "0.7", None, 1 + 0j])
def test_filter_rejects_invalid_threshold_types(threshold):
    with pytest.raises(TypeError):
        filter_detections_by_confidence(make_detections(), threshold)


def test_filter_does_not_mutate_original_collection():
    detections = make_detections()
    original = detections.copy()

    filtered = filter_detections_by_confidence(detections, 0.70)

    assert detections == original
    assert filtered is not detections
