"""
算法组合模块
支持多种增强方法的组合（串行叠加 或 并行加权）
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


def enhance_image_combined(image, methods, model=None, device=None, weights=None, **kwargs):
    """
    组合多种增强方法

    Args:
        image: numpy array, 输入图像
        methods: list of str, 增强方法列表 ('clahe', 'gamma', 'zero_dce')
        model: Zero-DCE模型
        device: torch设备
        weights: list of float (可选), 对应方法的权重。
                 如果提供，则并行处理并加权融合 (0.8*A + 0.1*B...)
                 如果不提供，则串行处理 (C(B(A(x))))
        **kwargs: 其他参数 (clahe_params, gamma_params)

    Returns:
        enhanced_image: numpy array, 增强后的图像
    """

    # --- 模式 1: 加权并行融合 (Parallel Weighted Fusion) ---
    if weights is not None:
        if len(methods) != len(weights):
            raise ValueError(f"methods数量 ({len(methods)}) 与 weights数量 ({len(weights)}) 不一致")

        # 使用 float32 累加器避免溢出
        h, w, c = image.shape
        accumulated_img = np.zeros((h, w, c), dtype=np.float32)

        for method, weight in zip(methods, weights):
            # 关键：每次都对【原始图像】image 进行处理
            if method == 'zero_dce':
                res = apply_zero_dce(image, model, device)
            elif method == 'clahe':
                clahe_params = kwargs.get('clahe_params', {})
                res = apply_clahe(image, **clahe_params)
            elif method == 'gamma':
                gamma_params = kwargs.get('gamma_params', {})
                res = apply_gamma(image, **gamma_params)
            else:
                raise ValueError(f"Unknown enhancement method: {method}")

            # 加权累加
            accumulated_img += res.astype(np.float32) * weight

        # 结果截断并转回 uint8
        enhanced_image = np.clip(accumulated_img, 0, 255).astype(np.uint8)
        return enhanced_image

    # --- 模式 2: 串行处理 (Sequential Processing) ---
    else:
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
