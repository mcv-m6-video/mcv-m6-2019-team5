from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from methods import week2_adaptive
from metrics import mean_average_precision
from model import Video
from operations.gaussian_model import get_background_model


def w2_map_alpha(alpha, rho=0.4):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    frames = []
    ious = []
    for im, mask, frame in week2_adaptive(video, alpha, rho, disable_tqdm=True):
        frames.append(frame)
        ious.append(frame.get_detection_iou_mean(ignore_classes=True))

    mAP = mean_average_precision(frames)

    print('alpha', alpha, 'rho', rho, 'mAP', mAP, 'mean IoU', np.mean(ious))

    return mAP


def main():
    alpha_values = np.linspace(1.5, 3, 20)
    rho_values = np.logspace(-2, -0.1, 20)

    # Ensure cache
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25),
                         disable_tqdm=False)

    # Best alpha: 1.75

    # mAP_list = Parallel(n_jobs=4)(delayed(w2_map_alpha)(alpha) for alpha in tqdm(alpha_values))
    mAP_list = Parallel(n_jobs=3)(delayed(w2_map_alpha)(1.75, rho) for rho in tqdm(rho_values))

    """mAP_list = [0.18656629994209614, 0.23257430508572247, 0.2333781161367368, 0.18301435406698566,
                0.1773032336790726, 0.1762025561112319, 0.12792207792207794, 0.17066218427456575,
                0.12438077386530996, 0.12091293755609694, 0.11872632575757576, 0.1189064558629776,
                0.15132634758802985, 0.157589106928314, 0.26284443191338397, 0.39380709780347006,
                0.43192630414348515, 0.357941584643725, 0.3186317361126976, 0.20596422790608496]"""

    plt.figure()
    plt.plot(rho_values, mAP_list)
    plt.xlabel(r'$\rho$ threshold')
    plt.ylabel('mAP')
    plt.show()


if __name__ == '__main__':
    main()
