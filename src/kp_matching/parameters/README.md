# parameters

## Overview

This directory defines configuration dataclasses used by keypoint matching.

## Components

| Component | Description |
| --------- | ----------- |
| [`common.py`](./common.py) | Common matching settings including matcher method, descriptor source method, and distance norm selection |
| [`ratio_test.py`](./ratio_test.py) | Ratio-test configuration and validation |
| [`flann/`](./flann/) | FLANN-specific index and matcher parameter models |
