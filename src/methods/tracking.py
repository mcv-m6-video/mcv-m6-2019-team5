import sys

from tqdm import tqdm
import motmetrics as mm

from model import Video
from utils import read_detections, read_annotations
from metrics import idf1


def tracking(optical_flow_method, tracking_method, debug, **kwargs):
    video = Video('../datasets/AICity_data/train/S03/c010/frames')
    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')
    gt = read_annotations('../datasets/AICity_data/train/S03/c010/m6-full_annotation.xml')
    acc = mm.MOTAccumulator(auto_id=True)
    start = 500
    previous_frame = None
    for i, frame in tqdm(enumerate(video.get_frames(start=start)), total=len(video) - start, file=sys.stdout):
        if i == 0:
            tracking_method(optical_flow_method, None, None, frame, detections[i + start], debug)
            previous_frame = frame
        elif i % 2 == 0:
            tracking_method(optical_flow_method, previous_frame, detections[i - 1 + start], frame, detections[i + start], debug)
            previous_frame = frame

    # idf1(detections[i], gt[i], acc)
    if not debug:
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        print(summary)
