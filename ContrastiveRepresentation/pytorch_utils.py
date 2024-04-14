import torch
import numpy as np

# 'cuda' device for supported NVIDIA GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'\
        if torch.backends.mps.is_available() else 'cpu')

def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert numpy array to torch tensor and send it to the device.

    Args:
        x (np.ndarray): Input numpy array.
        dtype (torch.dtype): Data type of the resulting torch tensor (default: torch.float32).

    Returns:
        torch.Tensor: Converted torch tensor.
    """
    tensor = torch.tensor(x, dtype=dtype, device=device)
    return tensor

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Args:
        x (torch.Tensor): Input torch tensor.

    Returns:
        np.ndarray: Converted numpy array.
    """
    if x.device.type == 'cuda':
        x = x.cpu()  # Moving tensor to CPU if it's on GPU
    numpy_array = x.detach().numpy()
    return numpy_array
