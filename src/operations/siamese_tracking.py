import numpy as np
import torch
from PIL import Image
from torch import cuda
from torchvision import transforms

from model import Frame
from nn.siamese_net import SiameseNet


class SiameseTracking:

    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.valid_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = SiameseNet(16)
        if cuda.is_available():
            self.model = self.model.cuda()

        state_dict = torch.load('../weights/siamese.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, frame: Frame, debug=False):
        # TODO
        pass

    def predict(self, im1, im2) -> float:
        with torch.no_grad():
            im1 = self.valid_transform(im1)
            im2 = self.valid_transform(im2)
            if cuda.is_available():
                im1 = im1.cuda()
                im2 = im2.cuda()

            output1 = self.model.forward_single(im1)
            output2 = self.model.forward_single(im2)
            x = output1.cpu().numpy()
            y = output2.cpu().numpy()
            return np.linalg.norm(x - y)
