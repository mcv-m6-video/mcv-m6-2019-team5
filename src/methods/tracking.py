import sys

from tqdm import tqdm

from methods import read_detections
from model import Video


def tracking(optical_flow_method, tracking_method, debug, **kwargs):
    video = Video('../datasets/AICity_data/train/S03/c010/frames')
    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')

    previous_frame = None
    for i, frame in tqdm(enumerate(video.get_frames()), total=len(video), file=sys.stdout):
        if i == 0:
            tracking_method(optical_flow_method, frame, detections[i], None, None)
        else:
            tracking_method(optical_flow_method, frame, detections[i], previous_frame, detections[i - 1])
