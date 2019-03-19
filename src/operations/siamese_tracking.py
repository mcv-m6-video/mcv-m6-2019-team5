import sys
import torch

from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

from nn import SiameseDataset
from nn.contrastive_loss import ContrastiveLoss
from nn.siamese_net import SiameseNet


class SiameseTracking:

    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.valid_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def train(self, dataset_path: str):
        train_set = SiameseDataset(dataset_path, self.train_transform)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
        print(train_set)

        model = SiameseNet(16)
        if torch.cuda.is_available():
            model = model.cuda()

        criterion = ContrastiveLoss(margin=1.)
        optimizer = Adam(model.parameters())
        scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        for epoch in trange(1, 11, desc="Training", file=sys.stdout):
            scheduler.step()
            for batch_idx, data in tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, 10),
                                        total=len(train_loader),
                                        file=sys.stdout):
                (sample1, sample2), target = data
                if torch.cuda.is_available():
                    sample1 = sample1.cuda()
                    sample2 = sample2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output1, output2 = model(sample1, sample2)

                loss = criterion(output1, output2, target)
                loss.backward()
                optimizer.step()

    def predict(self, im1, im2):
        pass
