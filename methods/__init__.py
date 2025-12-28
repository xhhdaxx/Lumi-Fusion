"""
Methods package
"""
from .clahe import apply_clahe
from .gamma import apply_gamma
from .fusion import apply_zero_dce, enhance_image_combined

__all__ = ['apply_clahe', 'apply_gamma', 'apply_zero_dce', 'enhance_image_combined']
