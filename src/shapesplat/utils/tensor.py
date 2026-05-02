import torch


def to_device_tree(x, device: torch.device):
    """把嵌套结构中的 tensor 移到指定设备；用于保持 pipeline 代码简洁。"""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device_tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_tree(v, device) for v in x)
    return x
