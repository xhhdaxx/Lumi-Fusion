"""
算法组合模块
支持多种增强方法的组合
"""
import numpy as np
import torch
from .clahe import apply_clahe
from .gamma import apply_gamma


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


def enhance_image_combined(image, methods, model=None, device=None, **kwargs):
    """
    组合多种增强方法
    
    Args:
        image: numpy array, 输入图像
        methods: list of str, 增强方法列表，按顺序应用 ('clahe', 'gamma', 'zero_dce')
        model: Zero-DCE模型（如果methods中包含'zero_dce'）
        device: torch设备（如果methods中包含'zero_dce'）
        **kwargs: 其他参数
            - clahe_params: dict, CLAHE参数 {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}
            - gamma_params: dict, Gamma参数 {'gamma': 2.2}
        
    Returns:
        enhanced_image: numpy array, 增强后的图像
    """
    enhanced = image.copy()
    
    for method in methods:
        if method == 'zero_dce':
            enhanced = apply_zero_dce(enhanced, model, device)
        elif method == 'clahe':
            clahe_params = kwargs.get('clahe_params', {})
            enhanced = apply_clahe(enhanced, **clahe_params)
        elif method == 'gamma':
            gamma_params = kwargs.get('gamma_params', {})
            enhanced = apply_gamma(enhanced, **gamma_params)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
    
    return enhanced
