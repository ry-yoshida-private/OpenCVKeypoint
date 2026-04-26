from __future__ import annotations

from typing import Any, TypeAlias

from .detector import KeyPointDetector
from .result import KeyPointDetectionResult

KPDetector: TypeAlias = KeyPointDetector[Any, Any, Any, Any]
KPDetectionResult: TypeAlias = KeyPointDetectionResult[Any, Any]

__all__ = ["KPDetector", "KPDetectionResult"]
