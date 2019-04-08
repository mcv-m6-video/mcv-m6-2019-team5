import argparse
import os

import torch
from torch import cuda
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from nn.dataloader import Dataset, BalancedBatchSampler
from nn.loss import OnlineTripletLoss
from nn import get_transforms, fit
from nn.network import EmbeddingNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('--min-images', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dims', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    if cuda.is_available():
        print('Device: {}'.format(cuda.get_device_name(0)))

    train_transform, test_transform = get_transforms(args)

    train_set = Dataset(os.path.join(args.dataset_dir, 'train'), train_transform, min_images=args.min_images)
    train_batch_sampler = BalancedBatchSampler(train_set.targets, n_classes=10, n_samples=10)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=4)
    print(train_set)

    model = EmbeddingNet(args.dims)
    if cuda:
        model = model.cuda()
    print(model)

    criterion = OnlineTripletLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, None, model, criterion, optimizer, scheduler, args.epochs, cuda)

    torch.save(model.state_dict(), '../weights/siamese_w6.pth')


if __name__ == '__main__':
    main()
