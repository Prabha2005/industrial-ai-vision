"""Image and video utilities built on OpenCV."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Box = tuple[int, int, int, int]
Color = tuple[int, int, int]


def _validate_image(image: Image) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("image must be a non-empty grayscale or color array")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("color image must have 1, 3, or 4 channels")


def load_image(path: str | Path) -> Image:
    """Load an image from disk, raising an explicit error if it cannot be decoded."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def resize_image(
    image: Image,
    width: int | None = None,
    height: int | None = None,
) -> Image:
    """Resize an image; preserve aspect ratio when only one dimension is given."""
    _validate_image(image)
    if width is None and height is None:
        raise ValueError("width or height must be provided")
    for name, value in (("width", width), ("height", height)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    original_height, original_width = image.shape[:2]
    if width is None:
        target_width = max(1, round(original_width * height / original_height))
        target_height = height
    elif height is None:
        target_width = width
        target_height = max(1, round(original_height * width / original_width))
    else:
        target_width = width
        target_height = height

    return cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def apply_gaussian_blur(image: Image, kernel_size: int = 5) -> Image:
    """Apply Gaussian smoothing using an odd, positive square kernel."""
    _validate_image(image)
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def threshold_image(image: Image, threshold: int = 127) -> Image:
    """Convert a grayscale or BGR image into a binary threshold mask."""
    _validate_image(image)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 <= threshold <= 255
    ):
        raise ValueError("threshold must be an integer from 0 to 255")

    if image.ndim == 2:
        grayscale = image
    elif image.shape[2] == 1:
        grayscale = image[:, :, 0]
    elif image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    _, mask = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    return mask


def annotate_image(
    image: Image,
    box: Box,
    label: str,
    confidence: float | None = None,
    color: Color = (0, 255, 0),
) -> Image:
    """Draw one labeled bounding box on a copy of the input image."""
    _validate_image(image)
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("box coordinates must form a positive-area box")
    if confidence is not None:
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")

    output = image.copy()
    display_label = label if confidence is None else f"{label} {confidence:.2f}"
    draw_color: Color | int = color
    if image.ndim == 2 or image.shape[2] == 1:
        draw_color = int(
            cv2.cvtColor(
                np.array([[color]], dtype=np.uint8), cv2.COLOR_BGR2GRAY
            )[0, 0]
        )
    cv2.rectangle(output, (x1, y1), (x2, y2), draw_color, thickness=2)
    cv2.putText(
        output,
        display_label,
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        draw_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return output


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n_frames: int = 1,
) -> list[Path]:
    """Save every Nth video frame as a numbered JPEG and return its paths."""
    if (
        isinstance(every_n_frames, bool)
        or not isinstance(every_n_frames, int)
        or every_n_frames <= 0
    ):
        raise ValueError("every_n_frames must be a positive integer")

    source_path = Path(video_path)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {source_path}")

    try:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % every_n_frames == 0:
                frame_path = destination / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Could not write video frame: {frame_path}")
                saved_paths.append(frame_path)
            frame_index += 1
        return saved_paths
    finally:
        capture.release()
