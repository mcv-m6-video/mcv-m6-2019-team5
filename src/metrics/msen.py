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
    # sen = np.power(sen, 2)

    sen[idx_zeros] = 0

    if plot:
        plt.figure()
        plt.title('Histogram of errors')
        plt.hist(sen[sen > 0], 25)
        plt.show()

    return float(np.mean(sen))
