import numpy as np
import bob.ip.optflow.hornschunck


def horn_schunck(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    # flow = bob.ip.optflow.hornschunck.VanillaFlow(im1.shape) # GRAY
    flow = bob.ip.optflow.hornschunck.Flow(im1.shape)
    u, v = flow.estimate(alpha=200, iterations=20, image1=im1, image2=im2)
    return np.concatenate((u[..., None], v[..., None]), axis=2)
