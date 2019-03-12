import cv2
from typing import List
from model import Video, Rectangle, Frame, Result
from operations.find_boxes import find_boxes
from operations.gaussian_model import get_background_model, gaussian_model_adaptive
from operations.morphological_operations import closing, opening, dilate
from utils import read_detections


def week2_adaptive(video: Video) -> List[Result]:
    model_mean, model_std = get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25))
    ground_truth = read_detections('../datasets/AICity_data/train/S03/c010/gt/gt.txt')
    frames = []
    results = []
    frame_id = int(2141 * 0.25)
    roi = cv2.cvtColor(cv2.imread('../datasets/AICity_data/train/S03/c010/roi.jpg'), cv2.COLOR_BGR2GRAY)
    for mask in gaussian_model_adaptive(video, int(2141 * 0.25), model_mean, model_std, alpha=1.75, rho=0.4,
                                        total_frames=int(2141 * 0.75)):
        mask = mask & roi
        cv2.imshow('f', mask)
        cv2.waitKey()

        mask = opening(mask, 3)
        cv2.imshow('f', mask)
        cv2.waitKey()

        mask = closing(mask, 35)
        cv2.imshow('f', mask)
        cv2.waitKey()

        detections = find_boxes(mask)

        frame = Frame(frame_id)
        frame.detections = detections
        frame.ground_truth = ground_truth[frame_id]

        frames.append(frame)

        frame_id += 1

        iou = frame.get_detection_iou(ignore_classes=True)
        print(iou)

        results.append(frame.to_result(ignore_classes=True))

    return results
