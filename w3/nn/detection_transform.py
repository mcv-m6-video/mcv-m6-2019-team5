import torchvision
from torchvision.transforms import Compose, Resize, Pad, ToTensor
import numpy as np

from model import Detection


class DetectionTransform(Compose):

    def __init__(self, input_size=(1920, 1080), side: int = 416):
        self.scale = np.max(input_size) / side

        if input_size[0] > input_size[1]:
            self.pad = (0, int(input_size[0] - input_size[1]) // 2)
        else:
            self.pad = (int(input_size[1] - input_size[0]) // 2, 0)
        super().__init__([
            Pad(self.pad, fill=0),
            Resize((side, side)),
            ToTensor()
        ])

    def __call__(self, im):
        a = super().__call__(im)
        return a

    def shrink_detection(self, det: Detection) -> None:
        top_left = (int((det.top_left[0] + self.pad[0]) / self.scale),
                    int((det.top_left[1] + self.pad[1]) / self.scale))
        det.top_left = top_left
        det.width = int(det.width / self.scale)
        det.height = int(det.height / self.scale)

    def unshrink_detection(self, det: Detection) -> None:
        top_left = (int(det.top_left[0] * self.scale - self.pad[0]),
                    int(det.top_left[1] * self.scale - self.pad[1]))
        det.top_left = top_left
        det.width = int(det.width * self.scale)
        det.height = int(det.height * self.scale)
