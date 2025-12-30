"""
Gamma校正算法
"""
import numpy as np


def apply_gamma(image, gamma=2.2):
    """
    应用Gamma校正算法
    
    Args:
        image: numpy array, shape (H, W, 3) 或 (H, W), 值范围 [0, 255] 或 [0, 1]
        gamma: float, gamma值，>1变亮，<1变暗
        
    Returns:
        enhanced_image: numpy array, 增强后的图像，uint8格式，值范围[0, 255]
    """
    # 确保是float类型
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    # 归一化到[0, 1]范围
    if image.max() > 1.0:
        image = image / 255.0

    # 应用gamma校正
    enhanced_image = np.power(image, 1.0 / gamma)

    # 限制到[0, 1]范围
    enhanced_image = np.clip(enhanced_image, 0.0, 1.0)

    # 转换回[0, 255]范围
    enhanced_image = (enhanced_image * 255.0).astype(np.uint8)

    return enhanced_image
