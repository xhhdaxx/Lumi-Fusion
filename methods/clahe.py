"""
CLAHE增强算法
"""
import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用CLAHE (Contrast Limited Adaptive Histogram Equalization) 算法
    
    Args:
        image: numpy array, shape (H, W, 3) 或 (H, W), 值范围 [0, 255]
        clip_limit: float, CLAHE的对比度限制参数
        tile_grid_size: tuple, 网格大小，用于局部直方图均衡化
        
    Returns:
        enhanced_image: numpy array, 增强后的图像，uint8格式
    """
    # 确保是uint8格式
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 如果是灰度图
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)
    # 如果是彩色图
    else:
        # 转换到LAB颜色空间，在L通道上应用CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        enhanced_image = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)
    
    return enhanced_image
