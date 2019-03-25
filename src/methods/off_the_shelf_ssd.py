import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from torch import cuda
from torchvision import transforms

from model import Video, Frame, Detection
from nn.ssd.ssd import build_ssd
from operations import KalmanTracking


def off_the_shelf_ssd(debug=False, *args):
    if cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    trans = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    labels = (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    model = build_ssd('test', 300, 21)  # initialize SSD
    model.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
    if torch.cuda.is_available():
        model = model.cuda()
    kalman = KalmanTracking()
    model.eval()
    with torch.no_grad():
        for i, im in enumerate(video.get_frames()):
            im_tensor = trans(im)
            im_tensor = im_tensor.view((-1,) + im_tensor.size())
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            output = model.forward(im_tensor)
            detections = output.data

            w = im.width
            h = im.height
            frame = Frame(i)

            # skip j = 0, because it's the background class
            for j in (2, 6, 7, 14):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                for cls_det in cls_dets:
                    x1 = int(w * cls_det[0])
                    y1 = int(h * cls_det[1])
                    det = Detection(-1, labels[j - 1], (x1, y1), width=w * (cls_det[2] - cls_det[0]),
                                    height=h * (cls_det[3] - cls_det[1]), confidence=cls_det[4])
                    frame.detections.append(det)

            kalman(frame)

            if debug:
                plt.figure()
                for det in frame.detections:
                    rect = patches.Rectangle(det.top_left, det.width, det.height,
                                             linewidth=2, edgecolor='blue', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(det.top_left[0], det.top_left[1], s='{} ~ {}'.format(det.label, det.id),
                             color='white', verticalalignment='top',
                             bbox={'color': 'blue', 'pad': 0})
                plt.imshow(im)
                plt.axis('off')
                # plt.savefig('../video/video_ssd_KalmanID/frame_{:04d}'.format(i))
                plt.show()
                plt.close()

