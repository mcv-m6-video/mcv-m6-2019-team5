import numpy as np
import torch
from torch import cuda

from model import Frame
from nn import get_transforms
from nn.network import EmbeddingNet


class SiameseDB:
    def __init__(self, dimensions: int, weights_path):
        self.dimensions = dimensions
        self.classes = []
        self.db = np.empty((0, dimensions))
        self.temp_classes = []
        self.temp_db = np.empty((0, dimensions))
        self.model = EmbeddingNet(dimensions)
        if cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        _, self.test_transform = get_transforms(dimensions)

    def process_frame(self, frame: Frame):
        cropped_images = []
        for detection in frame.detections:
            xtl, ytl = detection.top_left
            w = detection.width
            h = detection.height
            self.temp_classes.append(detection.id)
            cropped_images.append(frame.image[ytl:ytl + h, xtl:xtl + w])
        cropped_images = self.test_transform(cropped_images)
        if cuda.is_available():
            cropped_images = cropped_images.cuda()
        embedding = self.model(cropped_images)
        self.temp_db = np.vstack(self.temp_db, embedding)

    def update_db(self):
        self.db = np.vstack(self.db, self.temp_db)
        self.classes.append(self.temp_classes)
        self.temp_db = np.empty((0, self.dimensions))
        self.temp_classes = []
