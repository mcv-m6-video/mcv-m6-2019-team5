import torch


def pairwise_distances(vectors):
    return -2 * vectors.mm(torch.t(vectors)) + \
           vectors.pow(2).sum(dim=1).view(1, -1) + \
           vectors.pow(2).sum(dim=1).view(-1, 1)
