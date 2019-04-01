from vidstab import VidStab
import matplotlib.pyplot as plt


def adam_spanbaauer(video_pth: str, debug: bool = False ):
    # Using defaults
    stabilizer = VidStab()
    stabilizer.stabilize(input_path=video_pth,
                         output_path='default_stable_video.avi')

    # Using a specific keypoint detector
    stabilizer = VidStab(kp_method='ORB')
    stabilizer.stabilize(input_path=video_pth,
                         output_path='adamOut/ORB_stable_video.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
    stabilizer.stabilize(input_path=video_pth,
                         output_path='adamOut/FAST_stable_video.avi')

    if debug:

        stabilizer.plot_trajectory()
        plt.show()

        stabilizer.plot_transforms()
        plt.show()


if __name__ == '__main__':
    adam_spanbaauer()
