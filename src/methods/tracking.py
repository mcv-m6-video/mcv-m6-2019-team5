import sys

from tqdm import tqdm

from model import Video
from utils import read_detections


def tracking(optical_flow_method, tracking_method, debug, **kwargs):
    video = Video('../datasets/AICity_data/train/S03/c010/frames')
    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')

    previous_frame = None
    for i, frame in tqdm(enumerate(video.get_frames(start=500)), total=len(video) - 500, file=sys.stdout):
        if i == 0:
            tracking_method(optical_flow_method, None, None, frame, detections[i], debug)
            previous_frame = frame
        elif i % 2 == 0:
            tracking_method(optical_flow_method, previous_frame, detections[i - 1], frame, detections[i], debug)
            previous_frame = frame
