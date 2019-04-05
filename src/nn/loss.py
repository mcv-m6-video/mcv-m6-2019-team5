import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils import pairwise_distances


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        distance = (output2 - output1).pow(2).sum(1)
        loss = 0.5 * ((1 - target).float() * distance.pow(2) +
                      (target).float() * F.relu(self.margin - distance).pow(2))
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss function.
    Reference: https://arxiv.org/pdf/1503.03832.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet loss function.
    Reference: https://arxiv.org/pdf/1703.07737.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, targets):
        cuda = embeddings.is_cuda
        if cuda:
            embeddings = embeddings.cpu()
            targets = targets.cpu()

        distance_matrix = pairwise_distances(embeddings)

        triplets = []
        for target in set(targets):
            target_mask = (targets == target)
            positive_indices = np.where(target_mask)[0]
            negative_indices = np.where(np.logical_not(target_mask))[0]
            for i in positive_indices:
                hard_positive = positive_indices[np.argmax(distance_matrix[i, j] for j in positive_indices)]
                hard_negative = negative_indices[np.argmin(distance_matrix[i, j] for j in negative_indices)]
                triplets.append([i, hard_positive, hard_negative])
        triplets = torch.LongTensor(np.array(triplets))

        if cuda:
            embeddings = embeddings.cuda()
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()
