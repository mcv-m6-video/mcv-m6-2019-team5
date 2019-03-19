import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from torchvision import transforms

from model import Video, Detection, Frame
from nn import DetectionTransform
from nn.yolo.utils import utils
from nn.yolo.models import Darknet
from operations import KalmanTracking


def off_the_shelf_yolo(debug=False):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    detection_transform = DetectionTransform()
    classes = utils.load_classes('../config/coco.names')

    model = Darknet('../config/yolov3.cfg')
    model.load_weights('../weights/yolov3.weights')
    if torch.cuda.is_available():
        model = model.cuda()

    kalman = KalmanTracking()

    model.eval()
    with torch.no_grad():
        for i, im in enumerate(video.get_frames()):
            im_tensor = detection_transform(im)

            im_tensor = im_tensor.view((-1,) + im_tensor.size())
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            detections = model.forward(im_tensor)
            detections = utils.non_max_suppression(detections, 80)

            frame = Frame(i)

            if debug:
                plt.figure()

            for d in detections[0]:
                bbox = d.cpu().numpy()
                det = Detection(-1, classes[int(d[6])], (bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                                height=bbox[3] - bbox[1], confidence=d[5])
                detection_transform.unshrink_detection(det)
                frame.detections.append(det)

            kalman(frame)

            if debug:
                for det in frame.detections:
                    rect = patches.Rectangle(det.top_left, det.width, det.height,
                                             linewidth=2, edgecolor='blue', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(det.top_left[0], det.top_left[1], s='{} - {}'.format(det.label, det.id),
                             color='white', verticalalignment='top',
                             bbox={'color': 'blue', 'pad': 0})
                plt.imshow(im)
                plt.show()
                plt.close()
