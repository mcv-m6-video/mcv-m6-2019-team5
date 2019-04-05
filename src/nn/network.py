import multiprocessing.dummy as mp

from torch import nn
from torchvision import models


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_dims)

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):

    def __init__(self, num_dims: int = 16):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes=2)

    def forward(self, x1, x2):
        output1, output2 = self.pool.map(self.get_embedding, [x1, x2])
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):

    def __init__(self, num_dims: int = 16):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes=3)

    def forward(self, x1, x2, x3):
        output1, output2, output3 = self.pool.map(self.get_embedding, [x1, x2, x3])
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
