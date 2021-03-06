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

    idx_zeros = gt[:, :, 2] == 0

    sen = np.linalg.norm(flow_uv - gt_uv, axis=2)
    sen[idx_zeros] = 0

    return float(np.sum(sen > th) / (flow.shape[0] * flow.shape[1] - np.sum(idx_zeros)))
