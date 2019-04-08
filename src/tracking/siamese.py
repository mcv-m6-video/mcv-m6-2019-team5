import numpy as np

class Siamese:

    def __init__(self, n_dims: int = 32):
        self.db = np.empty((0, n_dims))
