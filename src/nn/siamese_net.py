from torch import nn
from torchvision import models


class SiameseNet(nn.Module):

    def __init__(self, num_dims: int = 16):
        super(SiameseNet, self).__init__()

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_dims)

    def forward(self, x1, x2):
        output1 = self.model(x1)
        output2 = self.model(x2)
        return output1, output2

    def get_embedding(self, data):
        return self.model(data)
