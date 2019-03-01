import numpy as np
from math import sqrt


def msen(detection: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean squared error in non occluded areas
    :param detection
    :param gt
    :return:
    """
    # minimum error to consider
    th = 3

    detection_v = detection[:,:,0]
    detection_u = detection[:,:,1]
    
    gt_v = gt[:,:,0]
    gt_u = gt[:,:,1]
    gt_val = gt[:,:,2]

    difference_v = gt_v - detection_v
    difference_u = gt_u - detection_u
    
    idx_zeros = list(np.where(gt_val==0))

    err_v = (difference_v * difference_v) / detection_v.size
    err_u = (difference_u * difference_u) / detection_u.size
    sen = sqrt(err_v * err_v + err_u * err_u)
    sen[idx_zeros] = 0
    non_error_pxls = list(np.where(sen <= th))
    sen[non_error_pxls] = 0

    msen = sum(sen)/sen.size

    return msen


