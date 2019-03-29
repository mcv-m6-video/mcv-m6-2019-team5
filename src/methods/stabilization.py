import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import Video


def stabilization(optical_flow_method, debug: bool = False, **kwargs):
    """
    Perform video stabilization using the given optical flow method.

    Idea: test some metric using a known logo. Using ORB matching we could detect if it moves.

    :param optical_flow_method: the optical flow method to use
    :param debug: whether to show debug plots
    """
    video = Video('../datasets/stabilization')

    previous_frame = None
    accum_flow = np.zeros(2)
    for frame in tqdm(video.get_frames(), total=len(video), file=sys.stdout):
        rows, cols, _ = frame.shape
        if previous_frame is not None:
            flow = optical_flow_method(previous_frame, frame)
            accum_flow += np.mean(flow, axis=(0, 1))

            transform = np.float32([[1, 0, accum_flow[0]], [0, 1, accum_flow[1]]])
            frame = cv2.warpAffine(frame, transform, (cols, rows))

            if debug:
                plt.figure()
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

        previous_frame = frame
