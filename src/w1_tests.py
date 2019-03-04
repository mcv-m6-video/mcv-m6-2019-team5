import cv2
import matplotlib.pyplot as plt
from functional import seq
from matplotlib.lines import Line2D

from metrics import msen, show_optical_flow, pepn, iou_over_time, mean_average_precision
from model import Video
from utils import read_detections, read_optical_flow, alter_detections

amount_frames = 40
make_video = False


def main():
    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi",
                  "../datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml",
                  car_only=False)
    """
        YOLO3 DETECTIONS
    """
    det_algs = ['mask_rcnn', 'ssd512', 'yolo3']
    for alg in det_algs:
        detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_{0}.txt'.format(alg))
        frames = []

        # roi = cv2.imread('../datasets/AICity_data/train/S03/c010/roi.jpg')

        for im, f in seq(video.get_frames()).take(40):
            f.detections = detections[f.id]
            frames.append(f)

            if make_video:
                make_video_frame(im, f, frames)

        iou_over_time(frames)

    """
        DETECTIONS FROM ALTERED GROUND TRUTH 
    """
    frames = []

    for im, f in seq(video.get_frames()).take(40):
        f.detections = alter_detections(f.ground_truth)
        frames.append(f)

        if make_video:
            make_video_frame(im, f, frames)

    iou_over_time(frames)

    """
        OPTICAL FLOW 
    """
    of_det_1 = read_optical_flow('../datasets/optical_flow/detection/LKflow_000045_10.png')
    of_det_2 = read_optical_flow('../datasets/optical_flow/detection/LKflow_000157_10.png')

    of_gt_1 = read_optical_flow('../datasets/optical_flow/gt/000045_10.png')
    of_gt_2 = read_optical_flow('../datasets/optical_flow/gt/000157_10.png')

    msen_45 = msen(of_det_1, of_gt_1, plot=True)
    pepn_45 = pepn(of_det_1, of_gt_1)
    print("Sequence 045: MSEN", msen_45, "PEPN", pepn_45)

    msen_157 = msen(of_det_2, of_gt_2, plot=True)
    pepn_157 = pepn(of_det_2, of_gt_2)
    print("Sequence 157: MSEN", msen_157, "PEPN", pepn_157)

    show_optical_flow(of_gt_1)


def make_video_frame(im, frame, frames):
    im1 = im.copy()

    for d in frame.ground_truth:
        cv2.rectangle(im1, (int(d.top_left[0]), int(d.top_left[1])),
                      (int(d.get_bottom_right()[0]), int(d.get_bottom_right()[1])), (255, 0, 0), thickness=5)

    for d in frame.detections:
        cv2.rectangle(im1, (int(d.top_left[0]), int(d.top_left[1])),
                      (int(d.get_bottom_right()[0]), int(d.get_bottom_right()[1])), (0, 0, 255), thickness=5)

    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.legend([
        Line2D([0], [0], color=(0, 0, 1)),
        Line2D([0], [0], color=(1, 0, 0)),
    ], ['Ground truth', 'Detection'])

    plt.subplot(2, 1, 2)
    plt.plot(range(frame.id + 1),
             seq(frames)
             .map(lambda fr: fr.get_detection_iou_mean())
             .to_list(),
             'b-', label='IoU'
             )
    plt.plot(range(frame.id + 1),
             seq(range(frame.id + 1))
             .map(lambda i: mean_average_precision(seq(frames).take(i).to_list()))
             .to_list(),
             'r-', label='mAP'
             )
    axes = plt.gca()
    axes.set_xlim((0, amount_frames))
    axes.set_ylim((0, 1))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
