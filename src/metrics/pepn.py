import numpy as np


def pepn(flow: np.ndarray, gt: np.ndarray) -> float:
    """
    Percentage of erroneous pixels in non occluded areas
    :param flow:
    :param gt:
    :return:
    """
    # minimum error to consider
    th = 3

    flow_uv = flow[:, :, 0:2]
    gt_uv = gt[:, :, 0:2]

    gt_val = gt[:, :, 2]
    idx_zeros = gt_val == 0

    err = np.linalg.norm(flow_uv - gt_uv, axis=2)
    sen = np.power(err, 2)
    sen[idx_zeros] = 0

    return float(sen[sen > th].size / (flow.size - np.sum(idx_zeros)))
