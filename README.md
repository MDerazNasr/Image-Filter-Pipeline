# C++ Image/Video Pipeline (Custom Filters)

A C++17 desktop project that implements an image/video processing pipeline from scratch:
- Grayscale (BGR → 1-channel)
- Box Blur (fast sliding-window implementation)
- Sobel edge detection

OpenCV is used **only** for loading/saving images and video (IO). All filtering math is custom C++.

## Features
- CPU single-thread mode
- CPU multithread mode (`std::thread`, row/column partitioning)
- Video processing with reusable buffers
- Per-stage timing (grayscale/blur/sobel) + FPS reporting

## Build (macOS)
Install dependencies:
```bash
brew install cmake opencv


In README, add:

```md
## Performance (M4 MacBook Pro)
Example video run (CPU multithread, 8 threads):
- ~6.6 ms/frame
- ~122 FPS average