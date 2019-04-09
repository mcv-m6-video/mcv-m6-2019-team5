import numpy as np


class SiameseDB:
    def __init__(self, dimensions: int):
        self.db = np.empty((0, dimensions))

    def __call__(self, *args, **kwargs):
        pass