# detectors

## Overview

This directory contains various keypoint detection algorithms implemented as Python classes. <br>
Each detector extends the base `KPDetector` class and provides different approaches to feature detection in images.

## Components

| Component | Description |
| --------- | ----------- |
| [agast.py](./agast.py) | AGAST (Adaptive and Generic Accelerated Segment Test) detector |
| [fast.py](./fast.py) | FAST (Features from Accelerated Segment Test) detector for corner detection |
| [harris.py](./harris.py) | Harris corner detector |
| [mser.py](./mser.py) | MSER (Maximally Stable Extremal Regions) detector for blob-like feature detection |
| [shi_tomashi.py](./shi_tomashi.py) | Shi-Tomasi corner detector (goodFeaturesToTrack) with quality-based corner selection |
| [simple_blob.py](./simple_blob.py) | Simple blob detector for detecting blob-like features in images |
| [standard.py](./standard.py) | Standard keypoint detector supporting ORB, SIFT, BRISK, AKAZE, and KAZE algorithms |
