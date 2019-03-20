import copy

import torch
from typing import List

from torch.utils.data import Dataset
from torchvision.transforms import Compose, functional

from model import Video, Detection
import numpy as np

from nn import DetectionTransform

MAX_DETECTIONS = 50


class YoloDataset(Dataset):

    def __init__(self, video: Video, gt: List[List[Detection]], classes: List[str],
                 transforms: DetectionTransform = None):
        self.video = video
        self.gt = gt
        self.classes = classes
        self.transforms = transforms

    def __getitem__(self, index):
        im = self.video[index]
        if self.transforms is not None:
            im = self.transforms(im)

        gt = self.gt[index]
        target = np.zeros((MAX_DETECTIONS, 5))
        for i, det in enumerate(gt):
            det = copy.copy(det)
            self.transforms.shrink_detection(det)
            target[i, :] = np.array([(det.top_left[0] + det.width // 2) / 416,
                                     (det.top_left[1] + det.height // 2) / 416,
                                     det.width / 416,
                                     det.height / 416,
                                     self.classes.index(det.label)])

        return im, torch.from_numpy(target)

    def __len__(self):
        return len(self.video)
