import numpy as np
import torch
from torch import cuda
from sklearn.neighbors import NearestNeighbors
from model import Frame, Detection
from nn import get_transforms
from nn.network import EmbeddingNet
from PIL import Image

THRESHOLD = 1


class SiameseDB:
    def __init__(self, dimensions: int, weights_path: str):
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
        self.model.eval()
        _, self.test_transform = get_transforms(dimensions)

    def process_frame(self, frame: Frame):
        cropped_images = []
        for detection in frame.detections:
            xtl, ytl = detection.top_left
            w = detection.width
            h = detection.height
            self.temp_classes.append(detection.id)
            cropped_image = frame.image[ytl:ytl + h, xtl:xtl + w]
            cropped_image = self.test_transform(Image.fromarray(cropped_image))
            cropped_images.append(cropped_image)
        if cuda.is_available():
            cropped_images = torch.cat(cropped_images, 0)
            cropped_images = cropped_images.cuda()
        embedding = self.model(cropped_images)
        self.temp_db = np.vstack(self.temp_db, embedding)

    def update_db(self):
        self.db = np.vstack(self.db, self.temp_db)
        self.classes.append(self.temp_classes)
        self.temp_db = np.empty((0, self.dimensions))
        self.temp_classes = []

    def query(self, image: np.ndarray, detection: Detection) -> int:
        xtl, ytl = detection.top_left
        w = detection.width
        h = detection.height
        cropped_image = image[ytl:ytl + h, xtl:xtl + w]
        cropped_image = self.test_transform(Image.fromarray(cropped_image))
        if cuda.is_available():
            cropped_image = torch.cat(cropped_image, 0)
            cropped_image = cropped_image.cuda()
        embedding = self.model(cropped_image)

        return self._get_class(embedding)

    def _get_class(self, embedding, k: int = 1) -> int:
        embedding = embedding.cpu().numpy()
        neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.db)
        distances, indices = neighbors.kneighbors(embedding)
        if distances[0, 0] > THRESHOLD:
            return -1
        return self.classes[indices[0, 0]]



