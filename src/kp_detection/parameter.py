from __future__ import annotations
import warnings
import cv2
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field
from .method import KPDetectionMethod

if TYPE_CHECKING:
    from .type_alias import KPDetector

@dataclass
class KPDetectionParameters:
    """
    Base parameters for keypoint detection (subclasses add method-specific fields).

    Attributes:
    ----------
    method: KPDetectionMethod
        The method for keypoint detection.
    is_brief_applied: bool
        Whether to use brief descriptor extractor.
    """
    method: KPDetectionMethod = KPDetectionMethod.SIFT
    is_brief_applied: bool = False
    brief: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Post-initialization validation.
        """
        self._validate()
        if self.is_brief_applied:
            self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # type: ignore

    def _validate(self) -> None:
        """
        Validate the parameters.
        """
        if self.is_brief_applied and not self.method.is_brief_supported():
            brief_methods = "\n".join(
                f"  - {m.value}" for m in KPDetectionMethod if m.is_brief_supported()
            )
            warnings.warn(
                "BRIEF descriptor extractor is not supported for "
                f"{self.method.value}. is_brief_applied is set to False.\n"
                "BRIEF is supported for:\n"
                f"{brief_methods}"
            )
            self.is_brief_applied = False

    def build_detector(self) -> KPDetector:
        """
        Build a keypoint detector instance based on the parameters.

        For SHI_TOMASHI use ShiTomashiParameters; for HARRIS use HarrisParameters.
        Calling this on base KPDetectionParameters with those methods raises ValueError.
        Other methods use this class as-is.

        Returns:
        ----------
        BuiltFromParametersKPDetector
            Detector instance for methods handled by KPDetectionParameters.
            Shi-Tomasi and Harris are built from ShiTomashiParameters and HarrisParameters.
        """

        if self.method in [KPDetectionMethod.SHI_TOMASHI, KPDetectionMethod.HARRIS]:
            raise ValueError(f"Invalid method: {self.method}, Shi-Tomasi and Harris are built from ShiTomashiParameters and HarrisParameters.")
        detector_class = self.method.detector_class
        return detector_class(params=self)
    


