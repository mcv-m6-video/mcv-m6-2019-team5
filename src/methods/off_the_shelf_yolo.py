import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from torchvision import transforms
from tqdm import tqdm

from model import Video, Detection, Frame
from nn import DetectionTransform
from nn.yolo.utils import utils
from nn.yolo.models import Darknet
from operations import KalmanTracking, OverlapTracking

VALID_LABELS = [1, 2, 3, 5, 7]


def off_the_shelf_yolo(tracking, debug=False, *args, **kwargs):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    detection_transform = DetectionTransform()
    classes = utils.load_classes('../config/coco.names')

    model = Darknet('../config/yolov3.cfg')
    model.load_weights('../weights/fine_tuned_yolo_freeze.weights')
    if torch.cuda.is_available():
        model = model.cuda()

    frames = []
    last_im = None

    model.eval()
    with torch.no_grad():
        for i, im in tqdm(enumerate(video.get_frames(start=200)), total=len(video), file=sys.stdout, desc='Yolo'):
            im_tensor = detection_transform(im)

            im_tensor = im_tensor.view((-1,) + im_tensor.size())
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            detections = model.forward(im_tensor)
            detections = utils.non_max_suppression(detections, 80, conf_thres=.75, nms_thres=0.2)

            frame = Frame(i)

            for d in detections[0]:
                if int(d[6]) in VALID_LABELS:
                    bbox = d.cpu().numpy()
                    det = Detection(-1, classes[int(d[6])], (bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                                    height=bbox[3] - bbox[1], confidence=d[5])
                    detection_transform.unshrink_detection(det)
                    frame.detections.append(det)

            if tracking is not None:
                last_frame = None if len(frames) == 0 else frames[-1]
                tracking(frame=frame, im=im, last_frame=last_frame, last_im=last_im, frames=frames, debug=False)

            frames.append(frame)

            last_im = im

            if debug:
                plt.figure()
                for det in frame.detections:
                    rect = patches.Rectangle(det.top_left, det.width, det.height,
                                             linewidth=2, edgecolor='blue', facecolor='none')
                    plt.gca().add_patch(rect)
                    if tracking is None:
                        text = '{}'.format(det.label)
                    else:
                        text = '{} ~ {}'.format(det.label, det.id)
                    plt.text(det.top_left[0], det.top_left[1], s=text,
                             color='white', verticalalignment='top',
                             bbox={'color': 'blue', 'pad': 0})
                plt.imshow(im)
                plt.axis('off')
                # plt.savefig('../video/video_yolo_fine_tune_good/frame_{:04d}'.format(i))
                plt.show()
                plt.close()
