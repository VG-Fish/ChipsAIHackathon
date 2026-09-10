import torch


def get_device() -> torch.device:
    """CUDA (Windows/Linux+NVIDIA) > MPS (Apple Silicon) > CPU, whichever is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
