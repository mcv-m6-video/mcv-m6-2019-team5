import cv2

from metrics import msen, pepn
from utils import show_optical_flow_arrows, read_optical_flow


def optical_flow(optical_flow_method, debug: bool = False, **kwargs):
    """
    Perform the optical flow using the given method and print metrics

    :param optical_flow_method: the optical flow method to use
    :param debug: whether to show debug plots
    """

    im1 = cv2.imread('../datasets/optical_flow/img/000157_10.png')  # 045
    im2 = cv2.imread('../datasets/optical_flow/img/000157_11.png')
    gt = read_optical_flow('../datasets/optical_flow/gt/000157_10.png')

    flow = optical_flow_method(im1, im2)

    if flow.shape[0] == gt.shape[0]:
        print('MSEN: ', msen(flow, gt, debug=debug))
        print('PEPN: ', pepn(flow, gt))

    if debug:
        show_optical_flow_arrows(im1, flow)
