import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def msen(flow: np.ndarray, gt: np.ndarray, debug: bool = False) -> float:
    """
    Mean squared error in non occluded areas
    """

    flow_uv = flow[:, :, 0:2]
    gt_uv = gt[:, :, 0:2]

    idx_zeros = gt[:, :, 2] == 0

    sen = np.linalg.norm(flow_uv - gt_uv, axis=2)

    if debug:
        sns.set(color_codes=True)
        plt.title('Histogram of errors')
        sns.distplot(sen[np.logical_not(idx_zeros)], bins=25, kde=False)

    return float(np.mean(sen[np.logical_not(idx_zeros)]))
