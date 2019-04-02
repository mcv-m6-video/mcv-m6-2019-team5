import sys

from tqdm import tqdm

from model import Video
from utils import read_detections, read_annotations
from metrics import Mot

START = 500
STRIDE = 1


def tracking(optical_flow_method, tracking_method, debug, **kwargs):
    video = Video('../datasets/AICity_data/train/S03/c010/frames')
    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')
    gt = read_annotations('../datasets/AICity_data/train/S03/c010/m6-full_annotation.xml')

    mot = Mot()

    previous_frame = None
    for i, frame in tqdm(enumerate(video.get_frames(start=START)), total=len(video) - START, file=sys.stdout):
        if i == 0:
            tracking_method(optical_flow_method, None, None, frame, detections[i + START], debug)
            previous_frame = frame
        elif i % STRIDE == 0:
            tracking_method(optical_flow_method, previous_frame, detections[i - STRIDE + START], frame,
                            detections[i + START], debug)
            previous_frame = frame

    idf1 = mot.get_idf1()
    print(idf1)
    a = mot.get_events()
    print(a)
