import sys

from joblib import Parallel, delayed

from methods import week2_nonadaptive
from model import Video
from metrics import mean_average_precision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from operations.gaussian_model import get_background_model


def w2_map_alpha(alpha):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    frames = []
    ious = []
    for im, mask, frame in week2_nonadaptive(video, alpha, disable_tqdm=True):
        frames.append(frame)
        ious.append(frame.get_detection_iou_mean(ignore_classes=True))

    mAP = mean_average_precision(frames)

    print('alpha', alpha, 'mAP', mAP, 'mean IoU', np.mean(ious))

    return mAP


def main():
    alpha_values = np.linspace(1.5, 3, 20)

    # Ensure cache
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25),
                         disable_tqdm=False)

    mAP_list = Parallel(n_jobs=4)(delayed(w2_map_alpha)(alpha) for alpha in tqdm(alpha_values))
    # for alpha in tqdm(alpha_values, file=sys.stdout, desc='Global loop....'):
    #     mAP_list.append(w2_map_alpha(alpha))

    plt.figure()
    plt.plot(alpha_values, mAP_list)
    plt.xlabel(r'$\alpha$ threshold')
    plt.ylabel('mAP')
    plt.savefig('map_alpha.png')
    plt.show()


if __name__ == '__main__':
    main()
