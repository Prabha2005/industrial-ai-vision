# Industrial AI Vision

A software-only industrial computer vision learning project. It currently
demonstrates pretrained YOLO inference on images and prerecorded videos.
Webcam capture, dataset training, and region-based counting are not implemented.
The project does not claim deployment on physical edge hardware.

## Milestone 1: OpenCV fundamentals

The first module provides image loading, aspect-preserving resizing, Gaussian
blur, binary thresholding, image annotation, and sampled video-frame extraction.

### Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for milestone status.

## Optional YOLO object detection

The detection extra pins Ultralytics to `8.4.215`. For CPU inference, use the
CPU PyTorch index, then install the project extra. If the system drive is low
on space, point Python temporary files at a drive with more room:

```powershell
New-Item -ItemType Directory -Force -Path "D:\prabha\ai-vision-temp"
New-Item -ItemType Directory -Force -Path "D:\prabha\ai-vision-temp\pip-cache"
$env:TEMP = "D:\prabha\ai-vision-temp"
$env:TMP = "D:\prabha\ai-vision-temp"
.\.venv\Scripts\python.exe -m pip install --cache-dir "D:\prabha\ai-vision-temp\pip-cache" torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --cache-dir "D:\prabha\ai-vision-temp\pip-cache" -e ".[detection]"
```

Download the public sample image, then run the single-image demonstration:

```powershell
New-Item -ItemType Directory -Force -Path .\data\samples | Out-Null
curl.exe -fL "https://ultralytics.com/images/bus.jpg" -o .\data\samples\bus.jpg
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe examples\run_image_inference.py
```

The thread limits avoid OpenBLAS allocation failures observed in this
environment; they are session-only settings.

The script uses the existing `YoloDetector` CPU adapter and annotation helper.
It preserves the raw results, applies confidence filtering, then class-aware
Non-Maximum Suppression (NMS), and annotates only the final detections. The
confidence threshold defaults to `0.70`; set a different threshold with
`--confidence-threshold`, for example:

```powershell
.\.venv\Scripts\python.exe examples\run_image_inference.py --confidence-threshold 0.50
```

NMS uses an IoU threshold of `0.50` by default; configure it with
`--iou-threshold`. For the IoU threshold, only same-class detections whose IoU
is strictly greater than the threshold are suppressed, so detections exactly
at the threshold are retained. Both thresholds must be between `0.0` and
`1.0`, inclusive. The demo prints counts before filtering, after confidence
filtering, and after NMS, then saves the annotated copy to
`outputs\detect\bus_detected.jpg`. The verified default run reported 5 raw
detections, 4 after confidence filtering, and 4 after NMS; the person detection
at confidence `0.656` was excluded. NMS did not reduce the remaining count on
this sample because no same-class pair had IoU above `0.50`. It used:

- Ultralytics `8.4.215`
- PyTorch `2.15.0+cpu`; torchvision `0.29.2+cpu`
- NumPy `2.4.6`; OpenCV `5.1.0`
- CUDA available: `False`

The `yolo26n.pt` checkpoint is COCO-pretrained and was downloaded from the
official [Ultralytics v8.4.0 assets release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt).
The checkpoint is stored under `models\weights` and the sample is stored under
`data\samples`; both are Git-ignored. The four retained detections were one bus
and three people. This is an example run, not an accuracy evaluation.

The Ultralytics package and YOLO26 models are offered under AGPL-3.0 or an
Enterprise license. Review the [current license terms](https://www.ultralytics.com/license)
before publishing or using the project commercially. Verify the sample image's
own terms before redistributing it. Unit tests use fake model outputs and do
not load or download weights.

## Recorded video inference

The frame-by-frame pipeline keeps only the current frame in memory. Video
dimensions come from the first decoded frame; a valid source FPS is preserved,
otherwise output uses 30 FPS. Output supports `.mp4` and `.avi`, subject to
available OpenCV codecs.

OpenCV's `VideoCapture.read()` reports whether a frame was returned, but its
generic API does not reliably distinguish normal end-of-file from a decoding or
read failure across video backends. Accordingly, a failed read after processing
starts is treated as end-of-input. A truncated or damaged video may therefore
produce a partial output without a decoding error; the reported frame count is
the number of frames actually processed, not a validation of video completeness.

### Detection-only mode

This is the default mode. Each frame goes through `YoloDetector`, the project
confidence filter (default `0.70`), and project NMS (default IoU `0.50`).
Detection-only confidence and IoU options are not passed to the tracking mode.

```powershell
.\.venv\Scripts\python.exe examples\run_video_inference.py `
  --input-video .\data\samples\recorded_clip.mp4 `
  --output-video .\outputs\video\recorded_clip_detected.mp4
```

Change detection thresholds with `--confidence-threshold` and
`--iou-threshold`.

### ByteTrack mode

Pass `--tracking` to use the standalone `ByteTrackTracker` on each frame. One
tracker instance is created per video-processing call and reused across that
video's frames. It receives low-confidence detections through Ultralytics
tracking inference (default input confidence `0.10`); the project confidence
filter and project NMS do not run before or after tracking. The tracking
confidence input can be changed with `--tracking-confidence-threshold`.
Combining that flag with detection-only threshold flags is rejected.

```powershell
.\.venv\Scripts\python.exe examples\run_video_inference.py `
  --input-video .\data\samples\recorded_clip.mp4 `
  --output-video .\outputs\video\recorded_clip_tracked.mp4 `
  --tracking
```

Tracked annotations include the class name and tracking ID. Both modes use CPU
inference; no physical edge hardware is involved. Tracking IDs may change due
to occlusion, missed detections, or other scene conditions. The total tracked
detection count is a count across frames, not a count of unique objects. These
demonstrations do not measure tracking accuracy and do not establish industrial
PPE compliance detection.

Real-video tracking validation was completed on the following local,
Git-ignored samples. The user visually inspected the long-clip output and
confirmed that tracking is visible:

| Input video | Output video | Frames processed | Total tracked detections | Average processing time/frame | Approx. throughput |
|---|---|---:|---:|---:|---:|
| `data\samples\recorded_clip.mp4` | `outputs\video\recorded_clip_tracked.mp4` | 852 | 7,784 | 0.073 s | 13.7 FPS |
| `data\samples\short_clip.mp4` | Not recorded | 149 | 1,484 | 0.435 s | Not reported |

These are sample-run observations, not accuracy or unique-object measurements.

The reported average processing time includes inference, post-processing,
annotation, and writing, but excludes frame decoding. Detection-only model
initialization occurs before timing; the tracking model is lazy-loaded on the
first frame and that first-call load is included. The example does not claim
real-time performance. Automated tests use fake video capture, writer,
detector, and tracker objects and do not download model weights.
