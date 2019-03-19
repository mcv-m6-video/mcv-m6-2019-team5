from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        dist = (output2 - output1).pow(2).sum(1)
        loss = 0.5 * ((1 - target).float() * dist.pow(2) +
                      target.float() * F.relu(self.margin - dist).pow(2))

        return loss.mean()
