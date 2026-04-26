from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Sequence


from .parameters import (
    KPMatchCommonParameters, 
    RatioTestParameters, 
    FLANNParameters,
    )
from .method import KPMatchMethod


@dataclass
class KPMatchingParameters:
    common_params: KPMatchCommonParameters = field(default_factory=KPMatchCommonParameters)
    ratio_test_params: RatioTestParameters = field(default_factory=RatioTestParameters)
    flann_params: FLANNParameters = field(default_factory=FLANNParameters)

    def __post_init__(self):
        if self.common_params.knn == 1 and self.ratio_test_params.is_enabled:
            raise ValueError("RatioTest requires kNN > 1")

    def define_matcher(self) -> cv2.BFMatcher | cv2.FlannBasedMatcher:
        """
        Define the matcher based on the method.

        Returns:
        ----------
        cv2.BFMatcher | cv2.FlannBasedMatcher
            The matcher.
        """
        match self.common_params.method:
            case KPMatchMethod.BF | KPMatchMethod.KNN:
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.common_params.is_cross_check_enabled)
            case KPMatchMethod.FLANN:
                return self.flann_params.define_matcher()

    def define_matching_function(self) -> Callable[[np.ndarray, np.ndarray], Sequence[cv2.DMatch] | Sequence[Sequence[cv2.DMatch]]]:
        """
        Define the matching function based on the method.

        Returns:
        ----------
        Callable[[np.ndarray, np.ndarray], Sequence[cv2.DMatch] | Sequence[Sequence[cv2.DMatch]]]
            The matching function.
        """
        matcher = self.define_matcher()
        match self.common_params.method:
            case KPMatchMethod.BF :
                return lambda desc1, desc2: matcher.match(desc1, desc2)
            case KPMatchMethod.FLANN | KPMatchMethod.KNN:
                return lambda desc1, desc2: matcher.knnMatch(desc1, desc2, k=self.common_params.knn)

    @property
    def is_ratio_test_enabled(self) -> bool:
        """
        Whether the ratio test is enabled.

        Returns:
        ----------
        bool
            Whether the ratio test is enabled.
        """
        return self.ratio_test_params.is_enabled

    @property
    def ratio_test_threshold(self) -> float:
        """
        The threshold for the ratio test.

        Returns:
        ----------
        float
            The threshold for the ratio test.
        """
        return self.ratio_test_params.threshold
