import sys

import numpy as np
import pandas
import torch
from PIL import Image
from torch import cuda
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange, tqdm

from nn import SiameseDataset
from nn.contrastive_loss import ContrastiveLoss
from nn.siamese_net import SiameseNet


def siamese_train(epochs=25, *args):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = SiameseDataset('../datasets/siamese_data', train_transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)

    model = SiameseNet(16)
    if cuda.is_available():
        model = model.cuda()

    criterion = ContrastiveLoss(margin=1.)
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

    print('Starting training...')

    table = []
    with trange(1, epochs + 1, desc="Training", file=sys.stdout) as pbar0:
        for epoch in pbar0:
            scheduler.step()
            losses = []
            for batch_idx, data in tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, epochs),
                                        total=len(train_loader),
                                        file=sys.stdout):
                (sample1, sample2), target = data
                if cuda.is_available():
                    sample1 = sample1.cuda()
                    sample2 = sample2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output1, output2 = model(sample1, sample2)

                loss = criterion(output1, output2, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            mean_loss = np.mean(losses)
            pbar0.set_postfix(loss=mean_loss)
            table.append([epoch, mean_loss])

    df = pandas.DataFrame(table, columns=['epoch', 'mean_loss'])
    df.set_index('epoch', inplace=True)
    print('\nTraining stats:')
    print(df)

    print('Finished training')
    print('Saving weights')
    torch.save(model.state_dict(), '../weights/siamese.pth')
    print('Saved weights')
