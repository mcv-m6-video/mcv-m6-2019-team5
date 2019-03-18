import os
from typing import Iterator

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Video(Dataset):

    def __getitem__(self, index):
        if self.torch:
            sample = default_loader(self.files[index])
            if self.transforms is not None:
                return self.transforms(sample)
        else:
            return cv2.imread(self.files[index])

    def __init__(self, video_path: str, torch: bool = False, transforms=None):
        self.video_path = video_path
        self.torch = torch
        self.transforms = transforms

        self.files = list(map(lambda p: os.path.join(video_path, p), os.listdir(video_path)))

    def get_frames(self, start: int = 0, end: int = 2141) -> Iterator[np.ndarray]:
        for i in range(start, end):
            yield self[i]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return 'Video(path={})'.format(self.video_path)

