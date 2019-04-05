from vidstab import VidStab
import matplotlib.pyplot as plt
import cv2
import os

from utils import generate_frames


def adam_spanbaauer(video_pth: str, debug: bool = False):
    if os.path.exists('../.cache/default_stable_video.avi'):
        print("Cached Video")
        out = cv2.VideoCapture('../.cache/default_stable_video.avi')

    else:
        # Using defaults
        stabilizer = VidStab()
        stabilizer.stabilize(input_path=video_pth,
                             output_path='../.cache/default_stable_video.avi')

        # Using a specific keypoint detector
        stabilizer = VidStab(kp_method='ORB')
        stabilizer.stabilize(input_path=video_pth,
                             output_path='../.cache/ORB_stable_video.avi')

        # Using a specific keypoint detector and customizing keypoint parameters
        stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
        stabilizer.stabilize(input_path=video_pth,
                             output_path='../.cache/FAST_stable_video.avi')

        out = cv2.VideoCapture('../.cache/default_stable_video.avi')

        if debug:
            stabilizer.plot_trajectory()
            plt.show()

            stabilizer.plot_transforms()
            plt.show()
    generate_frames(out, "adam")



