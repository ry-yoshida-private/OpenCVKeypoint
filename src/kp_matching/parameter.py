from __future__ import annotations

import cv2
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence


from .parameters import (
    KPMatchCommonParameters, 
    RatioTestParameters, 
    FLANNParameters,
    FLANNIndexType,
    )
from .method import KPMatchMethod


@dataclass
class KPMatchingParameters:
    common_params: KPMatchCommonParameters
    ratio_test_params: RatioTestParameters = field(default_factory=RatioTestParameters)
    flann_params: FLANNParameters = field(default_factory=FLANNParameters)

    def __post_init__(self):
        if self.common_params.knn == 1 and self.ratio_test_params.is_enabled:
            raise ValueError("RatioTest requires kNN > 1")
        self._validate_flann_index_type()

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
                return cv2.BFMatcher(
                    self.common_params.distance_norm,
                    crossCheck=self.common_params.is_cross_check_enabled,
                )
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

    def _validate_flann_index_type(self) -> None:
        """
        Validate and align FLANN settings with descriptor type.
        """
        if self.common_params.method != KPMatchMethod.FLANN:
            return

        is_binary_descriptor = self.common_params.detection_method.is_binary_descriptor_supported
        is_float_descriptor = self.common_params.detection_method.is_float_descriptor_supported
        if not is_binary_descriptor and not is_float_descriptor:
            raise ValueError(
                f"{self.common_params.detection_method.value} does not provide descriptors for FLANN."
            )

        # Common practical default: LSH for binary descriptors, KDTree for float descriptors.
        expected_algorithm = (
            FLANNIndexType.LSH if is_binary_descriptor else FLANNIndexType.KDTREE
        )
        if self.flann_params.algorithm != expected_algorithm:
            warnings.warn(
                (
                    "FLANN algorithm is not aligned with descriptor type. "
                    f"algorithm={self.flann_params.algorithm.name}, "
                    f"detection_method={self.common_params.detection_method.value}. "
                    f"Using {expected_algorithm.name}."
                ),
                stacklevel=2,
            )
            self.flann_params.algorithm = expected_algorithm

        if self.flann_params.is_binary != is_binary_descriptor:
            warnings.warn(
                (
                    "FLANN binary mode is not aligned with descriptor type. "
                    f"is_binary={self.flann_params.is_binary}, "
                    f"detection_method={self.common_params.detection_method.value}. "
                    f"Using is_binary={is_binary_descriptor}."
                ),
                stacklevel=2,
            )
            self.flann_params.is_binary = is_binary_descriptor