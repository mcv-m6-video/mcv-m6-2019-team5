import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import Video
from utils import show_optical_flow_arrows, generate_frames


def stabilization(optical_flow_method, debug: bool = False, **kwargs):
    """
    Perform video stabilization using the given optical flow method.

    Idea: test some metric using a known logo. Using ORB matching we could detect if it moves.

    :param optical_flow_method: the optical flow method to use
    :param debug: whether to show debug plots
    """
    video = Video('../datasets/stabilization/piano')
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    previous_frame = None
    accum_flow = np.zeros(2)
    count = 0
    for i, frame in tqdm(enumerate(video.get_frames()), total=len(video), file=sys.stdout):
        rows, cols, _ = frame.shape
        if previous_frame is not None:
            if i % 4 ==0:
                p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), mask=None,
                                             **feature_params)
                flow = optical_flow_method(previous_frame, frame, p0)
                if debug:
                    show_optical_flow_arrows(previous_frame, flow)

                accum_flow += -np.mean(flow[np.logical_or(flow[:, :, 0] != 0, flow[:, :, 1] != 0)], axis=(0, 1))
                if np.isnan(accum_flow).any():
                    accum_flow = (0, 0)
                transform = np.float32([[1, 0, accum_flow[0]], [0, 1, accum_flow[1]]])
                frame2 = cv2.warpAffine(frame, transform, (cols, rows))

                if debug:
                    plt.figure()
                    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                cv2.imwrite("../video/block/OrigianlFrame%04d.jpg" % count, frame)  # save frame as JPEG file
                cv2.imwrite("../video/block/StabilizedFrame%04d.jpg" % count, frame2)  # save frame as JPEG file

                count += 1
        previous_frame = frame
