from typing import List

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from model import Video, Detection


class YoloDataset(Dataset):

    def __init__(self, video: Video, gt: List[List[Detection]], transforms: Compose = None):
        self.video = video
        self.gt = gt
        self.transforms = transforms

    def __getitem__(self, index):
        im = self.video[index]
        if self.transforms is not None:
            im = self.transforms(im)

        gt = self.gt[index]

    def __len__(self):
        return len(self.video)
