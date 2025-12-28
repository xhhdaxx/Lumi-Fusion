"""
Zero-DCE训练脚本
"""
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
from models.zero_dce import enhance_net_nopool
from models.zero_dce_loss import L_color, L_spa, L_exp, L_TV


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    # 自动选择设备（CUDA/MPS/CPU）
    device = get_device()
    
    # 只在CUDA可用时设置CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
    
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
    
    # pin_memory 主要用于 CUDA，MPS 和 CPU 不需要
    pin_memory = (device.type == 'cuda')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=pin_memory
    )

    loss_color = L_color()
    loss_spa = L_spa(device)
    loss_exp = L_exp(16, 0.6, device)
    loss_tv = L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    DCE_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device)
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            Loss_TV = 200*loss_tv(A)
            loss_spa_value = torch.mean(loss_spa(enhanced_image, img_lowlight))
            loss_col = 5*torch.mean(loss_color(enhanced_image))
            loss_exp_value = 10*torch.mean(loss_exp(enhanced_image))
            
            # best_loss
            loss = Loss_TV + loss_spa_value + loss_col + loss_exp_value

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="weight/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="weight/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
