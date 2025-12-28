"""
Evaluation package
"""
from .psnr import calculate_psnr
from .ssim import calculate_ssim
from .entropy import calculate_entropy


def calculate_all_metrics(img_ref, img_enhanced):
    """
    计算所有评估指标
    
    Args:
        img_ref: numpy array, 参考图像（原始低光图像或ground truth）
        img_enhanced: numpy array, 增强后的图像
        
    Returns:
        metrics: dict, 包含PSNR、SSIM和熵的字典
    """
    from .psnr import calculate_psnr
    from .ssim import calculate_ssim
    from .entropy import calculate_entropy
    
    psnr = calculate_psnr(img_ref, img_enhanced)
    ssim = calculate_ssim(img_ref, img_enhanced)
    entropy_value = calculate_entropy(img_enhanced)
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'Entropy': entropy_value
    }


__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_entropy', 'calculate_all_metrics']
