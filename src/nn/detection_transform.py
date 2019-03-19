from torchvision.transforms import Compose, Resize, Pad, ToTensor
import numpy as np

from model import Detection


class DetectionTransform(Compose):

    def __init__(self, input_size=(1920, 1080), side: int = 416):
        self.scale = np.max(input_size) / side
        self.small_x = int(input_size[0] / self.scale)
        self.small_y = int(input_size[1] / self.scale)
        self.pad_x = (side - self.small_x) // 2
        self.pad_y = (side - self.small_y) // 2
        super().__init__([
            Resize((self.small_y, self.small_x)),
            Pad((self.pad_x, self.pad_y)),
            ToTensor()
        ])

    def __call__(self, im):
        return super().__call__(im)

    def shrink_detection(self, det: Detection) -> None:
        top_left = (int(det.top_left[0] / self.scale + self.pad_x), int(det.top_left[1] / self.scale + self.pad_y))
        det.top_left = top_left
        det.width = int(det.width / self.scale)
        det.height = int(det.height / self.scale)

    def unshrink_detection(self, det: Detection) -> None:
        top_left = (int((det.top_left[0] - self.pad_x) * self.scale), int((det.top_left[1] - self.pad_y) * self.scale))
        det.top_left = top_left
        det.width = int(det.width * self.scale)
        det.height = int(det.height * self.scale)
