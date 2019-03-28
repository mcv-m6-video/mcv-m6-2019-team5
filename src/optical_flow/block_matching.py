import numpy as np


class BlockMatching:

    def __init__(self, block_size=8, window_size=64):
        self.block_side = block_size
        self.window_size = window_size

    def __call__(self, im: np.ndarray) -> np.ndarray:
        pass
