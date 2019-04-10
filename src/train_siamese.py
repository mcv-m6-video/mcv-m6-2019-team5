import argparse
import os
import time

import torch
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torch import cuda
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from nn import get_transforms, fit, extract_embeddings
from nn.dataloader import Dataset, BalancedBatchSampler
from nn.loss import OnlineTripletLoss
from nn.network import EmbeddingNet

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('validation_dir', type=str, nargs='?')
    parser.add_argument('--min-images', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dims', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    writer = SummaryWriter(
        log_dir='../runs/siamese_w6_{}_epochs_{}_dims_{}_{}'.format(os.path.basename(args.dataset_dir), args.epochs,
                                                                    args.dims, time.time()))

    if cuda.is_available():
        print('Device: {}'.format(cuda.get_device_name(0)))

    train_transform, test_transform = get_transforms(args)

    train_set = Dataset(args.dataset_dir, train_transform, min_images=args.min_images)
    train_batch_sampler = BalancedBatchSampler(train_set.targets, n_classes=10, n_samples=10)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=4)
    print(train_set)

    test_loader = None
    if args.validation_dir is not None:
        test_set = Dataset(args.validation_dir, transform=test_transform, min_images=args.min_images)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(test_set)

    model = EmbeddingNet(args.dims)
    if cuda:
        model = model.cuda()
    print(model)

    criterion = OnlineTripletLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda, writer=writer)

    train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
    writer.add_embedding(train_embeddings, metadata=train_targets, tag='Train embeddings')

    if test_loader is not None:
        test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)
        writer.add_embedding(test_embeddings, metadata=test_targets, tag='Test embeddings')

    print('Saving model...')
    torch.save(model.state_dict(),
               '../weights/siamese_w6_{}_epochs_{}_dims_{}.pth'.format(os.path.basename(args.dataset_dir), args.epochs,
                                                                       args.dims))
    print('Finished')


if __name__ == '__main__':
    main()
