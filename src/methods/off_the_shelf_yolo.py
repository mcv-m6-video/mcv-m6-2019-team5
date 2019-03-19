import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from torchvision import transforms

from model import Video, Detection, Frame
from nn.yolo.utils import utils
from nn.yolo.models import Darknet


def off_the_shelf_yolo(debug=False):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    trans = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
    classes = utils.load_classes('../config/coco.names')

    model = Darknet('../config/yolov3.cfg')
    model.load_weights('../weights/yolov3.weights')
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        for i, im in enumerate(video.get_frames()):
            im_tensor = trans(im)
            im_tensor = im_tensor.view((-1,) + im_tensor.size())
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            detections = model.forward(im_tensor)
            detections = utils.non_max_suppression(detections, 80)
            im_show = np.copy(im)

            scale_x = im.width / 416
            scale_y = im.height / 416

            frame = Frame(i)

            if debug:
                plt.figure()

            for d in detections[0]:
                bbox = d.cpu().numpy()
                x1 = int(scale_x * bbox[0])
                y1 = int(scale_y * bbox[1])
                det = Detection(-1, classes[int(d[6])], (x1, y1), width=scale_x * (bbox[2] - bbox[0]),
                                height=scale_y * (bbox[3] - bbox[1]), confidence=d[5])
                frame.detections.append(det)
                if debug:
                    rect = patches.Rectangle((x1, y1), det.width, det.height,
                                             linewidth=2, edgecolor='blue', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(x1, y1, s=det.label,
                             color='white', verticalalignment='top',
                             bbox={'color': 'blue', 'pad': 0})

            if debug:
                plt.imshow(im_show)
                plt.show()
                plt.close()
