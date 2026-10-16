import numpy as np
gpu_enable = True
try:
    import mlx.core as mx
    mlx = mx
except ImportError:
    gpu_enable = False
from dezero import Variable