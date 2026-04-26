from __future__ import annotations

from ..result import KPT, KPElem, KeyPointDetectionResult, KPDetectionStep
from .array import ArrayKPDetectionResult
from .cv2_keypoint import KPDetectionResult

KeyPointDetectionResult = KPDetectionResult | ArrayKPDetectionResult

__all__ = [
    "KPT",
    "KPElem",
    "KeyPointDetectionResult",
    "KPDetectionStep",
    "KPDetectionResult",
    "ArrayKPDetectionResult",
]
