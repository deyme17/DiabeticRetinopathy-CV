import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def calculate_mean_std(dataset: Dataset, batch_size: int = 64) -> tuple[list[float], list[float]]:
    """
    Calculate mean and std for dataset normalization.
    Args:
        dataset: PyTorch Dataset (should return tensors in [0, 1] range)
        batch_size: Batch size for calculation
    Returns:
        tuple: (mean, std) as lists for each channel [R, G, B]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0
    
    for images, _ in tqdm(loader, desc="Processing batches"):
        # images shape: [batch, channels, height, width]
        channels_sum += images.mean([0, 2, 3])
        channels_squared_sum += (images ** 2).mean([0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    return mean_list, std_list