from typing import List

import numpy as np
import torch
from PIL import Image
from torch import cuda
from torchvision import transforms

from model import Frame
from nn.siamese_net import SiameseNet
from utils import IDGenerator
from utils.crop_image import crop_image
from functional import seq


class SiameseTracking:

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.valid_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = None

    def _init_model(self):
        self.model = SiameseNet(16)
        if cuda.is_available():
            self.model = self.model.cuda()

        state_dict = torch.load('../weights/siamese.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, frame: Frame, im: Image, last_frame: Frame, last_im: Image, debug=False, *args,
                 **kwargs):
        if self.model is None:
            self._init_model()

        if last_frame is not None:
            ims1 = [crop_image(im, rectangle) for rectangle in frame.detections]
            ims2 = [crop_image(last_im, rectangle) for rectangle in last_frame.detections]

            embeddings1 = [self._get_embedding(im1) for im1 in ims1]
            embeddings2 = [self._get_embedding(im2) for im2 in ims2]
            for i, embedding1 in enumerate(embeddings1):
                min_j, min_distance = (seq(enumerate(embeddings2))
                                       .map(lambda pair: (pair[0], np.linalg.norm(embedding1 - pair[1])))
                                       .min_by(lambda pair: pair[1]))
                if min_distance < self.threshold:
                    frame.detections[i].id = last_frame.detections[min_j].id

        for detection in frame.detections:
            if detection.id == -1:
                detection.id = IDGenerator.next()

    def _get_embedding(self, im1) -> np.ndarray:
        with torch.no_grad():
            im1 = self.valid_transform(im1)
            if cuda.is_available():
                im1 = im1.cuda()

            output1 = self.model.get_embedding(im1.reshape((1,) + im1.size()))
            return output1.cpu().numpy()
