import warnings
from dataclasses import dataclass

from ..method import KPMatchMethod

@dataclass
class KPMatchCommonParameters:
    """
    Common parameters for keypoint matching.

    Attributes:
    ----------
    method: KPMatchMethod
        The method for keypoint matching.
    is_cross_check_enabled: bool
        Whether to use cross-check matching.
    knn: int
        The number of nearest neighbors to use for matching.
    """
    method: KPMatchMethod = KPMatchMethod.KNN
    is_cross_check_enabled: bool = False
    knn: int = 2

    def __post_init__(self):
        if self.knn < 1:
            raise ValueError("knn must be at least 1")
        if self.knn > 2:
            raise ValueError("knn must be less than or equal to 2")
        if self.method == KPMatchMethod.KNN and self.is_cross_check_enabled:
            warnings.warn("kNN matching cannot be used with cross-check")
            self.is_cross_check_enabled = False