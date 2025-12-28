import torch

def get_device():
    """
    自动选择设备：优先CUDA，其次MPS，最后CPU
    Returns:
        torch.device: 选择的设备
    """
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


