import cv2
from functional import seq

from metrics import msen, show_optical_flow, pepn, iou_over_time
from model import Video
from utils import read_detections, read_optical_flow


def main():
    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi",
                  "../datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml",
                  car_only=True)

    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')
    frames = []

    # roi = cv2.imread('../datasets/AICity_data/train/S03/c010/roi.jpg')

    for im, f in seq(video.get_frames()).take(40):
        f.detections = detections[f.id]
        frames.append(f)

        """""for d in detections[f.id]:
    
    
            cv2.rectangle(im, d.top_left, d.get_bottom_right(), (255, 0, 0))

        im = cv2.bitwise_and(im, roi)
        
        cv2.imshow('frame', im)
        cv2.waitKey()"""
    iou_over_time(frames)



    of_det_1 = read_optical_flow('../datasets/optical_flow/detection/LKflow_000045_10.png')
    of_det_2 = read_optical_flow('../datasets/optical_flow/detection/LKflow_000157_10.png')

    of_gt_1 = read_optical_flow('../datasets/optical_flow/gt/000045_10.png')
    of_gt_2 = read_optical_flow('../datasets/optical_flow/gt/000157_10.png')

    msen_of = msen(of_det_2, of_gt_2)
    pepn_of = pepn(of_det_2, of_gt_2)

    print(msen_of, pepn_of)
    show_optical_flow(of_det_1)


if __name__ == '__main__':
    main()
