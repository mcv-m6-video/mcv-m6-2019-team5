import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from model import Video
from ssd.ssd import build_ssd


def off_the_shelf_ssd(debug=False):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    trans = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    model = build_ssd('test', 300, 21)  # initialize SSD
    model.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
    if torch.cuda.is_available():
        model = model.cuda()

    v2 = Video("../datasets/AICity_data/train/S03/c010/frames")

    model.eval()
    with torch.no_grad():
        for im in video.get_frames():
            im = im.view((-1,) + im.size())
            if torch.cuda.is_available:
                im = im.cuda()

            output = model.forward(im)
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            plt.imshow(cv2.cvtColor(v2[0], cv2.COLOR_BGR2RGB))  # plot the image for matplotlib
            currentAxis = plt.gca()
            scale = 1
            labels = (  # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
            detections = output.data
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    score = detections[0, i, j, 0]
                    label_name = labels[i - 1]
                    display_txt = '%s: %.2f' % (label_name, score)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    color = colors[i]
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                    j += 1

            plt.show()
