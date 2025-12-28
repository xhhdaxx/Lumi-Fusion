"""
图像增强算法模块
实现CLAHE和Gamma校正算法
"""
import cv2
import numpy as np
from PIL import Image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用CLAHE (Contrast Limited Adaptive Histogram Equalization) 算法
    
    Args:
        image: numpy array, shape (H, W, 3) 或 (H, W), 值范围 [0, 255]
        clip_limit: float, CLAHE的对比度限制参数
        tile_grid_size: tuple, 网格大小，用于局部直方图均衡化
        
    Returns:
        enhanced_image: numpy array, 增强后的图像
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


def apply_zero_dce(image, model, device):
    """
    应用Zero-DCE模型进行增强
    
    Args:
        image: numpy array, shape (H, W, 3), 值范围 [0, 255]
        model: Zero-DCE模型
        device: torch设备
        
    Returns:
        enhanced_image: numpy array, 增强后的图像，uint8格式，值范围[0, 255]
    """
    import torch
    
    # 确保是RGB格式，如果是RGBA则转换为RGB
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    # 归一化到[0, 1]并转换为tensor
    if image.max() > 1.0:
        image_normalized = image.astype(np.float32) / 255.0
    else:
        image_normalized = image.astype(np.float32)
    
    image_tensor = torch.from_numpy(image_normalized)
    
    # 转换为CHW格式
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        _, enhanced_image_tensor, _ = model(image_tensor)
    
    # 转换回numpy
    enhanced_image = enhanced_image_tensor.squeeze(0).cpu().numpy()
    enhanced_image = enhanced_image.transpose(1, 2, 0)
    
    # 限制到[0, 1]范围
    enhanced_image = np.clip(enhanced_image, 0, 1)
    
    # 转换到[0, 255]
    enhanced_image = (enhanced_image * 255.0).astype(np.uint8)
    
    return enhanced_image


def enhance_image(image, method, model=None, device=None, **kwargs):
    """
    统一的图像增强接口
    
    Args:
        image: numpy array, 输入图像
        method: str, 增强方法 ('clahe', 'gamma', 'zero_dce')
        model: Zero-DCE模型（仅在method='zero_dce'时需要）
        device: torch设备（仅在method='zero_dce'时需要）
        **kwargs: 其他参数
        
    Returns:
        enhanced_image: numpy array, 增强后的图像
    """
    if method == 'clahe':
        return apply_clahe(image, **kwargs)
    elif method == 'gamma':
        return apply_gamma(image, **kwargs)
    elif method == 'zero_dce':
        return apply_zero_dce(image, model, device)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


def enhance_image_combined(image, methods, model=None, device=None, **kwargs):
    """
    组合多种增强方法
    
    Args:
        image: numpy array, 输入图像
        methods: list of str, 增强方法列表，按顺序应用
        model: Zero-DCE模型（如果methods中包含'zero_dce'）
        device: torch设备（如果methods中包含'zero_dce'）
        **kwargs: 其他参数
        
    Returns:
        enhanced_image: numpy array, 增强后的图像
    """
    enhanced = image.copy()
    
    for method in methods:
        if method == 'zero_dce':
            enhanced = apply_zero_dce(enhanced, model, device)
        elif method == 'clahe':
            enhanced = apply_clahe(enhanced, **kwargs.get('clahe_params', {}))
        elif method == 'gamma':
            enhanced = apply_gamma(enhanced, **kwargs.get('gamma_params', {}))
    
    return enhanced

