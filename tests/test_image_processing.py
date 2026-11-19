import cv2
import numpy as np
import pytest

from industrial_vision.image_processing import (
    annotate_image,
    apply_gaussian_blur,
    load_image,
    resize_image,
    threshold_image,
)

IMAGE_PROCESSORS = [
    pytest.param(lambda image: resize_image(image, width=4), id="resize"),
    pytest.param(lambda image: apply_gaussian_blur(image), id="gaussian-blur"),
    pytest.param(lambda image: threshold_image(image), id="threshold"),
    pytest.param(
        lambda image: annotate_image(image, (0, 0, 2, 2), "object"),
        id="annotate",
    ),
]


@pytest.mark.parametrize("process_image", IMAGE_PROCESSORS)
def test_image_processing_accepts_uint8_input(process_image):
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    processed = process_image(image)

    assert processed.dtype == np.uint8


@pytest.mark.parametrize("process_image", IMAGE_PROCESSORS)
def test_image_processing_rejects_float32_input(process_image):
    image = np.zeros((4, 4, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype uint8"):
        process_image(image)


def test_load_image_reads_color_image(tmp_path):
    source = np.full((12, 20, 3), 80, dtype=np.uint8)
    image_path = tmp_path / "source.png"
    assert cv2.imwrite(str(image_path), source)

    loaded = load_image(image_path)

    assert loaded.shape == (12, 20, 3)
    assert loaded.dtype == np.uint8


def test_load_image_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not read image"):
        load_image(tmp_path / "missing.png")


def test_resize_image_preserves_aspect_ratio_for_single_dimension():
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    resized = resize_image(image, width=10)

    assert resized.shape == (5, 10, 3)


def test_resize_image_requires_a_dimension():
    with pytest.raises(ValueError, match="width or height"):
        resize_image(np.zeros((4, 4), dtype=np.uint8))


def test_resize_image_rejects_zero_width():
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(image, width=0)


def test_gaussian_blur_rejects_even_kernel():
    with pytest.raises(ValueError, match="positive odd"):
        apply_gaussian_blur(np.zeros((5, 5), dtype=np.uint8), kernel_size=4)


def test_gaussian_blur_smooths_an_impulse_without_changing_dimensions():
    image = np.zeros((11, 11), dtype=np.uint8)
    image[5, 5] = 255

    blurred = apply_gaussian_blur(image, kernel_size=5)

    assert blurred.shape == image.shape
    assert blurred[5, 5] < 255
    assert np.count_nonzero(blurred) > 1


def test_threshold_image_separates_dark_and_bright_pixels():
    image = np.array([[10, 200]], dtype=np.uint8)

    mask = threshold_image(image, threshold=100)

    np.testing.assert_array_equal(mask, np.array([[0, 255]], dtype=np.uint8))


def test_threshold_image_converts_bgr_to_grayscale():
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

    mask = threshold_image(image, threshold=100)

    np.testing.assert_array_equal(mask, np.array([[0, 255]], dtype=np.uint8))


def test_annotate_image_draws_without_mutating_input():
    image = np.zeros((24, 24, 3), dtype=np.uint8)

    annotated = annotate_image(image, (2, 2, 18, 18), "worker", 0.9)

    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(image) == 0


@pytest.mark.parametrize("shape", [(24, 24), (24, 24, 1)])
def test_annotate_image_draws_on_grayscale_input(shape):
    image = np.zeros(shape, dtype=np.uint8)

    annotated = annotate_image(image, (2, 2, 18, 18), "worker", 0.9)

    assert annotated.shape == image.shape
    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(image) == 0
