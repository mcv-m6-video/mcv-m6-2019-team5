import numpy as np


def msen(flow: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean squared error in non occluded areas
    :param flow
    :param gt
    :return:
    """

    flow_uv = flow[:, :, 0:2]
    gt_uv = gt[:, :, 0:2]

    gt_val = gt[:, :, 2]
    idx_zeros = np.argwhere(gt_val == 0)

    err = np.linalg.norm(flow_uv - gt_uv, axis=2)
    sen = np.power(err, 2)

    sen[idx_zeros] = 0

    return float(np.mean(sen))
