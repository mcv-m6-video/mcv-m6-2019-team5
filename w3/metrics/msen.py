import numpy as np
import matplotlib.pyplot as plt


def msen(flow: np.ndarray, gt: np.ndarray, plot: bool = False) -> float:
    """
    Mean squared error in non occluded areas
    :param flow
    :param gt
    :return:
    """

    flow_uv = flow[:, :, 0:2]
    gt_uv = gt[:, :, 0:2]

    idx_zeros = gt[:, :, 2] == 0

    sen = np.linalg.norm(flow_uv - gt_uv, axis=2)

    if plot:
        plt.figure()
        plt.title('Histogram of errors')
        plt.hist(sen[np.logical_not(idx_zeros)], 25)
        plt.show()

    return float(np.mean(sen[np.logical_not(idx_zeros)]))
