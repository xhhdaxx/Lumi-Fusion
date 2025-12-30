"""
统一的实验测试脚本
支持所有7种实验配置：
1. CLAHE
2. Gamma
3. Zero-DCE
4. CLAHE + Zero-DCE
5. Gamma + Zero-DCE
6. Zero-DCE + Gamma
7. CLAHE + Zero-DCE + Gamma
"""
import os
import sys
import argparse
import glob
import numpy as np
from PIL import Image
import torch
import json
import csv

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.zero_dce import enhance_net_nopool
from methods.fusion import enhance_image_combined
from evaluation import calculate_all_metrics

# 实验配置定义
EXPERIMENTS = {
    '1': {
        'name': '01_CLAHE',
        'methods': ['clahe'],
        'description': '仅使用CLAHE算法'
    },
    '2': {
        'name': '02_Gamma',
        'methods': ['gamma'],
        'description': '仅使用Gamma校正'
    },
    '3': {
        'name': '03_Zero-DCE',
        'methods': ['zero_dce'],
        'description': '仅使用Zero-DCE模型'
    },
    '4': {
        'name': '04_CLAHE_Zero-DCE',
        'methods': ['clahe', 'zero_dce'],
        'description': '先CLAHE后Zero-DCE'
    },
    '5': {
        'name': '05_Gamma_Zero-DCE',
        'methods': ['gamma', 'zero_dce'],
        'description': '先Gamma后Zero-DCE'
    },
    '6': {
        'name': '06_Zero-DCE_Gamma',
        'methods': ['zero_dce', 'gamma'],
        'description': '先Zero-DCE后Gamma'
    },
    '7': {
        'name': '07_CLAHE_Zero-DCE_Gamma',
        'methods': ['clahe', 'zero_dce', 'gamma'],
        'description': 'CLAHE + Zero-DCE + Gamma'
    },
    '8': {
        'name': '08_Weighted_Fusion',
        'methods': ['clahe', 'gamma', 'zero_dce'],
        'weights': [0.8, 0.1, 0.1],  # 权重设置
        'description': '加权融合: 80% CLAHE + 10% Gamma + 10% Zero-DCE'
    }
}


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


def load_zero_dce_model(model_path, device):
    """加载Zero-DCE模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Zero-DCE model not found at: {model_path}")

    dce_net = enhance_net_nopool().to(device)
    dce_net.load_state_dict(torch.load(model_path, map_location=device))
    dce_net.eval()
    return dce_net


def process_image(image_path, experiment_config, dce_model, device,
                  clahe_params=None, gamma_params=None):
    """
    处理单张图像
    """
    # 读取图像
    image = np.array(Image.open(image_path))

    # 如果图像是RGBA，转换为RGB
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # 获取权重配置 (如果存在，则为加权融合模式)
    weights = experiment_config.get('weights', None)

    # 使用融合方法处理图像
    enhanced = enhance_image_combined(
        image,
        experiment_config['methods'],
        model=dce_model,
        device=device,
        clahe_params=clahe_params or {},
        gamma_params=gamma_params or {},
        weights=weights  # 传递权重参数
    )

    return enhanced


def run_experiment(exp_id, test_data_path, output_dir, model_path, device,
                   clahe_params=None, gamma_params=None):
    """
    运行单个实验
    """
    exp_config = EXPERIMENTS[exp_id]
    print(f"\n{'=' * 60}")
    print(f"Running Experiment {exp_id}: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    if 'weights' in exp_config:
        print(f"Fusion Weights: {dict(zip(exp_config['methods'], exp_config['weights']))}")
    print(f"{'=' * 60}\n")

    # 加载Zero-DCE模型（如果需要）
    dce_model = None
    if 'zero_dce' in exp_config['methods']:
        print(f"Loading Zero-DCE model from {model_path}")
        try:
            dce_model = load_zero_dce_model(model_path, device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # 创建输出目录
    exp_output_dir = os.path.join(output_dir, exp_config['name'])
    os.makedirs(exp_output_dir, exist_ok=True)

    # 获取所有测试图像
    all_images = []
    dataset_folders = []

    # 遍历测试数据目录
    if os.path.isdir(test_data_path):
        for folder_name in os.listdir(test_data_path):
            folder_path = os.path.join(test_data_path, folder_name)
            if os.path.isdir(folder_path):
                dataset_folders.append(folder_name)
                images = glob.glob(os.path.join(folder_path, "*"))
                images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                all_images.extend(images)

    if not all_images:
        print(f"No images found in {test_data_path}")
        return None

    print(f"Found {len(all_images)} images in {len(dataset_folders)} datasets")

    # 存储所有指标
    all_psnr = []
    all_ssim = []
    all_entropy = []

    # 处理每张图像
    for idx, image_path in enumerate(all_images):
        print(f"Processing [{idx + 1}/{len(all_images)}]: {os.path.basename(image_path)}")

        try:
            # 处理图像
            enhanced_image = process_image(
                image_path, exp_config, dce_model, device,
                clahe_params, gamma_params
            )

            # 保存结果
            rel_path = os.path.relpath(image_path, test_data_path)
            output_path = os.path.join(exp_output_dir, rel_path)
            output_dir_parent = os.path.dirname(output_path)
            os.makedirs(output_dir_parent, exist_ok=True)

            # 保存增强后的图像
            Image.fromarray(enhanced_image).save(output_path)

            # 计算指标（使用原始图像作为参考）
            original_image = np.array(Image.open(image_path))
            if len(original_image.shape) == 3 and original_image.shape[2] == 4:
                original_image = original_image[:, :, :3]

            # 计算指标
            metric_results = calculate_all_metrics(original_image, enhanced_image)

            all_psnr.append(metric_results['PSNR'])
            all_ssim.append(metric_results['SSIM'])
            all_entropy.append(metric_results['Entropy'])

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # 计算平均指标
    results = {
        'experiment_id': exp_id,
        'experiment_name': exp_config['name'],
        'num_images': len(all_images),
        'avg_PSNR': np.mean(all_psnr) if all_psnr else 0.0,
        'avg_SSIM': np.mean(all_ssim) if all_ssim else 0.0,
        'avg_Entropy': np.mean(all_entropy) if all_entropy else 0.0,
        'std_PSNR': np.std(all_psnr) if all_psnr else 0.0,
        'std_SSIM': np.std(all_ssim) if all_ssim else 0.0,
        'std_Entropy': np.std(all_entropy) if all_entropy else 0.0
    }

    print(f"\nExperiment {exp_id} Results:")
    print(f"  Average PSNR: {results['avg_PSNR']:.4f} ± {results['std_PSNR']:.4f}")
    print(f"  Average SSIM: {results['avg_SSIM']:.4f} ± {results['std_SSIM']:.4f}")
    print(f"  Average Entropy: {results['avg_Entropy']:.4f} ± {results['std_Entropy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run enhancement experiments')

    # 数据路径
    parser.add_argument('--test_data_path', type=str, default='data/test_data/',
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='results/images/',
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str, default='weight/Epoch99.pth',
                        help='Path to Zero-DCE model')

    # 实验选择 (包含8)
    parser.add_argument('--experiment_ids', type=str, nargs='+',
                        default=['1', '2', '3', '4', '5', '6', '7', '8'],
                        help='Experiment IDs to run (1-8)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments')

    # 算法参数
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit')
    parser.add_argument('--clahe_tile_size', type=int, default=8,
                        help='CLAHE tile grid size')
    parser.add_argument('--gamma_value', type=float, default=2.2,
                        help='Gamma correction value')

    # 其他
    parser.add_argument('--save_results_csv', type=str, default='results/metrics/experiment_results.csv',
                        help='Path to save results CSV file')
    parser.add_argument('--save_results_json', type=str, default='results/metrics/experiment_results.json',
                        help='Path to save results JSON file')

    args = parser.parse_args()

    # 确定要运行的实验
    if args.run_all:
        exp_ids = ['1', '2', '3', '4', '5', '6', '7', '8']
    else:
        exp_ids = args.experiment_ids

    # 设备
    device = get_device()

    # 算法参数
    clahe_params = {
        'clip_limit': args.clahe_clip_limit,
        'tile_grid_size': (args.clahe_tile_size, args.clahe_tile_size)
    }
    gamma_params = {
        'gamma': args.gamma_value
    }

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_results_csv) if args.save_results_csv else 'results/metrics/', exist_ok=True)

    # 运行所有实验
    all_results = []
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"Warning: Unknown experiment ID {exp_id}, skipping...")
            continue

        try:
            results = run_experiment(
                exp_id, args.test_data_path, args.output_dir, args.model_path, device,
                clahe_params, gamma_params
            )
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error running experiment {exp_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果到JSON
    if args.save_results_json:
        with open(args.save_results_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.save_results_json}")

    # 保存结果到CSV
    if args.save_results_csv and all_results:
        fieldnames = ['experiment_id', 'experiment_name', 'num_images',
                      'avg_PSNR', 'std_PSNR', 'avg_SSIM', 'std_SSIM',
                      'avg_Entropy', 'std_Entropy']
        with open(args.save_results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results saved to {args.save_results_csv}")

    # 打印汇总
    if all_results:
        print(f"\n{'=' * 60}")
        print("Summary of All Experiments")
        print(f"{'=' * 60}")
        print(f"{'Experiment':<25} {'PSNR':<12} {'SSIM':<12} {'Entropy':<12}")
        print("-" * 60)
        for result in all_results:
            print(f"{result['experiment_name']:<25} "
                  f"{result['avg_PSNR']:>8.4f}    "
                  f"{result['avg_SSIM']:>8.4f}    "
                  f"{result['avg_Entropy']:>8.4f}")
        print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
