"""
PSNR评估指标
"""
import numpy as np
try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None


def calculate_psnr(img1, img2, data_range=255.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: numpy array, 参考图像
        img2: numpy array, 增强后的图像
        data_range: float, 图像数据的范围（默认255）
        
    Returns:
        psnr: float, PSNR值
    """
    # 确保数据类型一致
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 如果图像值在[0, 1]范围，转换到[0, 255]
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0
        data_range = 255.0
    
    # 计算PSNR
    if peak_signal_noise_ratio is not None:
        psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)
    else:
        # Fallback implementation
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
    
    return psnr
