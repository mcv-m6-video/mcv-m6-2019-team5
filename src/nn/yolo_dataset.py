import torch
from typing import List

from torch.utils.data import Dataset
from torchvision.transforms import Compose, functional

from model import Video, Detection
import numpy as np

MAX_DETECTIONS = 50


class YoloDataset(Dataset):

    def __init__(self, video: Video, gt: List[List[Detection]], classes: List[str], transforms: Compose = None):
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
            target[i, :] = np.array([det.top_left[0], det.top_left[1], det.width, det.height,
                                     self.classes.index(det.label)])

        return im, functional.to_tensor(target)

    def __len__(self):
        return len(self.video)
