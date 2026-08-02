"""Small shared helpers that don't belong to any single module."""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Seed Python's random module, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
