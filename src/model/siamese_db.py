from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import cuda

from model import Frame, Detection
from nn import get_transforms
from nn.network import EmbeddingNet

THRESHOLD = 1


class SiameseDB:
    def __init__(self, dimensions: int, weights_path: str):
        self.dimensions = dimensions
        self.classes = []
        self.db = np.empty((0, dimensions))
        self.temp_classes = []
        self.temp_db = np.empty((0, dimensions))

        self.knn = None

        self.model = EmbeddingNet(dimensions)
        if cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()
        _, self.test_transform = get_transforms(dimensions)

    def process_frame(self, frame: Frame, detections: List[Detection]):
        if len(detections) == 0:
            return

        cropped_images_list = []
        for detection in detections:
            if detection.id == -1:
                continue
            xtl, ytl = detection.top_left
            w = detection.width
            h = detection.height
            self.temp_classes.append(detection.id)
            cropped_image = frame.image[ytl:ytl + h, xtl:xtl + w]
            cropped_image = self.test_transform(Image.fromarray(cropped_image))
            cropped_images_list.append(cropped_image)

        if len(cropped_images_list) == 0:
            return

        cropped_images_tensor = torch.stack(tuple(cropped_images_list), 0)
        if cuda.is_available():
            cropped_images_tensor = cropped_images_tensor.cuda()

        with torch.no_grad():
            embedding = self.model(cropped_images_tensor)
        self.temp_db = np.vstack((self.temp_db, embedding.cpu().numpy()))

    def update_db(self):
        self.db = np.vstack((self.db, self.temp_db))
        self.classes += self.temp_classes
        self.temp_db = np.empty((0, self.dimensions))
        self.temp_classes = []

        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.db)

    def query(self, image: np.ndarray, detection: Detection) -> int:
        if self.knn is None:
            return -1

        xtl, ytl = detection.top_left
        w = detection.width
        h = detection.height
        cropped_image = image[ytl:ytl + h, xtl:xtl + w]
        cropped_image = self.test_transform(Image.fromarray(cropped_image))
        cropped_image = torch.stack((cropped_image,), 0)
        if cuda.is_available():
            cropped_image = cropped_image.cuda()

        with torch.no_grad():
            embedding = self.model(cropped_image)

        return self._get_class(embedding)

    def _get_class(self, embedding: torch.Tensor) -> int:
        if self.db.shape[0] == 0:
            return -1
        embedding = embedding.cpu().numpy()
        distances, indices = self.knn.kneighbors(embedding)
        if distances[0, 0] > THRESHOLD:
            return -1
        return self.classes[indices[0, 0]]
