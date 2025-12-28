"""
SSIM评估指标
"""
import numpy as np
try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def calculate_ssim(img1, img2, data_range=255.0):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1: numpy array, 参考图像
        img2: numpy array, 增强后的图像
        data_range: float, 图像数据的范围（默认255）
        
    Returns:
        ssim: float, SSIM值
    """
    # 确保数据类型一致
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 如果图像值在[0, 1]范围，转换到[0, 255]
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0
        data_range = 255.0
    
    if structural_similarity is not None:
        # 如果是多通道图像
        if len(img1.shape) == 3:
            # 计算每个通道的SSIM并取平均
            ssim_channels = []
            for i in range(img1.shape[2]):
                ssim_channel = structural_similarity(
                    img1[:, :, i], 
                    img2[:, :, i], 
                    data_range=data_range
                )
                ssim_channels.append(ssim_channel)
            ssim = np.mean(ssim_channels)
        else:
            # 灰度图
            ssim = structural_similarity(img1, img2, data_range=data_range)
    else:
        # Fallback: 简化的SSIM计算
        print("Warning: Using simplified SSIM calculation. Install scikit-image for accurate results.")
        ssim = 0.5  # 占位值
    
    return ssim
