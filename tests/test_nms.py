import math

import pytest

from industrial_vision.detector import (
    Detection,
    intersection_over_union,
    non_max_suppression,
)


def make_detection(
    x1=0,
    y1=0,
    x2=10,
    y2=10,
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


def test_iou_of_identical_boxes_is_one():
    assert intersection_over_union(make_detection(), make_detection()) == 1.0


def test_iou_of_non_overlapping_boxes_is_zero():
    first = make_detection()
    second = make_detection(x1=10, x2=20)

    assert intersection_over_union(first, second) == 0.0


def test_iou_of_partially_overlapping_boxes():
    first = make_detection()
    second = make_detection(x1=5, x2=15)

    assert intersection_over_union(first, second) == pytest.approx(1 / 3)


def test_iou_rejects_non_detection_inputs():
    with pytest.raises(TypeError, match="Detection"):
        intersection_over_union(make_detection(), object())


def test_nms_suppresses_lower_confidence_overlapping_same_class():
    high_confidence = make_detection(confidence=0.95)
    low_confidence = make_detection(x1=1, y1=1, x2=11, y2=11, confidence=0.80)

    assert non_max_suppression([low_confidence, high_confidence]) == [
        high_confidence
    ]


def test_nms_retains_overlapping_detections_of_different_classes():
    first = make_detection(confidence=0.95)
    second = make_detection(class_id=1, confidence=0.80)

    assert non_max_suppression([first, second]) == [first, second]


def test_nms_handles_multiple_overlapping_detections():
    highest = make_detection(confidence=0.99)
    overlapping = make_detection(x1=1, y1=1, x2=11, y2=11, confidence=0.90)
    separate = make_detection(x1=30, y1=30, x2=40, y2=40, confidence=0.80)
    lower_overlapping = make_detection(
        x1=1.5,
        y1=1.5,
        x2=11.5,
        y2=11.5,
        confidence=0.70,
    )

    assert non_max_suppression(
        [lower_overlapping, separate, overlapping, highest]
    ) == [highest, separate]


def test_nms_retains_detection_at_iou_threshold_boundary():
    first = make_detection(confidence=0.95)
    second = make_detection(x1=5, x2=15, confidence=0.80)
    threshold = intersection_over_union(first, second)

    assert non_max_suppression([first, second], threshold) == [first, second]


def test_nms_accepts_empty_input():
    assert non_max_suppression([]) == []


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf])
def test_nms_rejects_invalid_iou_threshold_values(threshold):
    with pytest.raises(ValueError):
        non_max_suppression([], threshold)


@pytest.mark.parametrize("threshold", [True, False, "0.5", None, 1 + 0j])
def test_nms_rejects_invalid_iou_threshold_types(threshold):
    with pytest.raises(TypeError):
        non_max_suppression([], threshold)


def test_nms_does_not_mutate_input_collection():
    detections = [
        make_detection(confidence=0.8),
        make_detection(x1=20, x2=30, confidence=0.9),
    ]
    original = detections.copy()

    result = non_max_suppression(detections)

    assert detections == original
    assert result is not detections


def test_equal_confidence_detections_preserve_input_order_deterministically():
    first = make_detection(confidence=0.8)
    second = make_detection(x1=1, y1=1, x2=11, y2=11, confidence=0.8)

    result = non_max_suppression([first, second])

    assert result == [first]
    assert result[0] is first


def test_nms_rejects_non_detection_collection_items():
    with pytest.raises(TypeError, match="Detection"):
        non_max_suppression([make_detection(), object()])
