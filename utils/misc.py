import random
from typing import Union
from pathlib import Path
import os

import numpy as np

import torch


def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
def set_devices(devices: str):
    """
    Set cuda devices

    Args:
        devices (str): Comma separated string of device numbers. e.g., "0", "0, 1"
    """
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory

def load_causal_graph(path: str, normalize: bool = True) -> np.ndarray:
    if path.endswith('.pt'):
        cg = torch.load(path, map_location='cpu').float().numpy()
    elif path.endswith('.npy'):
        cg = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {path}")
    if normalize:
        cg_min, cg_max = cg.min(), cg.max()
        if cg_max > cg_min:
            cg = (cg - cg_min) / (cg_max - cg_min)
    return cg