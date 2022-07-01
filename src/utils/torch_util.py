import torch
import random
from typing import Optional
import numpy as np

def seed_everything(log, seed: Optional[int] = None, workers: bool = False) -> int:
    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f"set torch.backends.cudnn.benchmark=False")
    torch.backends.cudnn.benchmark=False
    log.info(f"set torch.backends.cudnn.deterministic=True")
    torch.backends.cudnn.deterministic = True