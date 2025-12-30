"""
信息熵评估指标
"""
import numpy as np


def calculate_entropy(image):
    """
    计算图像的信息熵
    
    Args:
        image: numpy array, 输入图像
        
    Returns:
        entropy_value: float, 信息熵值
    """
    # 转换为灰度图（如果是彩色图）
    if len(image.shape) == 3:
        # 使用RGB转灰度的标准公式
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()

    # 确保是uint8格式用于直方图计算
    if gray.max() <= 1.0:
        gray = (gray * 255.0).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)

    # 计算直方图 (256个bins，范围0-255)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))

    # 归一化直方图得到概率分布
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    else:
        return 0.0

    # 计算信息熵 (避免log(0))
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0

    entropy_value = -np.sum(hist * np.log2(hist))

    return entropy_value
