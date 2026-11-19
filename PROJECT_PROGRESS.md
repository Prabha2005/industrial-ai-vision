# Project progress

| Milestone | Status | Acceptance evidence |
|---|---|---|
| 1. OpenCV fundamentals | Completed | 15 focused tests passed at milestone completion; grayscale annotation supports 2D and single-channel images. A generated-video smoke test verifies frame extraction. Includes image I/O, resize, blur, thresholding, annotation, and sampled frame extraction. |
| 2. Object detection | Steps 1–5 complete; review pending | Confidence filtering then class-aware NMS. CPU bus-image demo reported 5 raw, 4 after confidence filtering, and 4 after NMS; no qualifying same-class overlap was suppressed. Full suite at milestone completion: 81 passed. Current full suite: 160 passed. Checkpoint, sample, and output are Git-ignored. |
| 3. Video inference and tracking | Video inference and ByteTrack integration implemented; real-video tracking validation recorded (Step 2F). Region-based counting is not implemented. | CPU ByteTrack runs: `recorded_clip.mp4` 852 frames, 7,784 tracked detections, 0.073 s/frame (~13.7 FPS); `short_clip.mp4` 149 frames, 1,484 tracked detections, 0.435 s/frame. Long-clip output visually inspected by the user. Counts are per-frame detections, not unique objects; no tracking accuracy or PPE-compliance evaluation claimed. |
| 4. Instance segmentation | Not started | |
| 5. Industrial use cases and dataset workflow | Not started | |
| 6. Model evaluation | Not started | |
| 7. FastAPI and SQLite | Not started | |
| 8. Monitoring dashboard | Not started | |
| 9. ONNX optimization and benchmarking | Not started | |
| 10. Docker and deployment documentation | Not started | |
