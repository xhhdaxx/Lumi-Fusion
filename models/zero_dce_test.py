"""
Zero-DCE测试脚本
"""
import torch
import torchvision
import os
import sys
import argparse
import time
import glob
from PIL import Image
import numpy as np

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.zero_dce import enhance_net_nopool


def get_device():
    """自动选择设备：优先CUDA，其次MPS，最后CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    return device


def lowlight(image_path, device, model_path='weight/Epoch99.pth', output_dir='data/result/'):
    """处理单张低光照图像"""
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    DCE_net = enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(f"Processing time: {end_time:.4f}s")

    # 构建输出路径
    rel_path = os.path.relpath(image_path, 'data/test_data/')
    result_path = os.path.join(output_dir, rel_path)
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='data/test_data/')
    parser.add_argument('--model_path', type=str, default='weight/Epoch99.pth')
    parser.add_argument('--output_dir', type=str, default='data/result/')
    args = parser.parse_args()

    # 自动选择设备（CUDA/MPS/CPU）
    device = get_device()

    with torch.no_grad():
        filePath = args.test_data_path
        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(f"Processing: {image}")
                lowlight(image, device, args.model_path, args.output_dir)
