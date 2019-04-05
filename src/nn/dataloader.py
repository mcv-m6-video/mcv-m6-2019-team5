import os

import numpy as np
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import ImageFolder

POSITIVE_NEGATIVE_RATIO = 0.1  # higher means more positives


class Dataset(ImageFolder):

    def __init__(self, root, transform, min_images=1):
        self.min_images = min_images
        super().__init__(root, transform)
        print('Found {} images belonging to {} classes'.format(len(self), len(self.classes)))

    def _find_classes(self, dir):
        classes, class_to_idx = super()._find_classes(dir)
        classes = [cls for cls in classes if len(os.listdir(os.path.join(dir, cls))) >= self.min_images]
        class_to_idx = {cls: idx for cls, idx in class_to_idx.items() if cls in classes}
        return classes, class_to_idx


class SiameseDataset(Dataset):
    """
    For each sample creates randomly a positive or a negative pair.
    """

    def __init__(self, root, transform, min_images=1):
        super().__init__(root, transform, min_images)

        self.label_to_idxs = {label: np.where(np.array(self.targets) == self.class_to_idx[label])[0] for label in
                              self.classes}
        np.random.seed(42)

    def __getitem__(self, index):
        target = int(np.random.random_sample() > POSITIVE_NEGATIVE_RATIO)
        if target == 0:
            siamese_label = self.classes[self.targets[index]]
        else:
            siamese_label = np.random.choice(list(set(self.classes) - {self.classes[self.targets[index]]}))
        siamese_index = np.random.choice(self.label_to_idxs[siamese_label])

        sample1 = self.loader(self.samples[index][0])
        sample2 = self.loader(self.samples[siamese_index][0])
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return (sample1, sample2), target


class TripletDataset(Dataset):
    """
    For each sample (anchor) randomly chooses a positive and negative samples.
    """

    def __init__(self, root, transform, min_images=1):
        super().__init__(root, transform, min_images)

        self.target_to_idxs = {target: np.where(np.array(self.targets) == target)[0] for target in
                               [self.class_to_idx[label] for label in self.classes]}
        np.random.seed(42)

    def __getitem__(self, index):
        anchor_target = self.targets[index]
        positive_target = anchor_target
        negative_target = np.random.choice(list({self.class_to_idx[label] for label in self.classes} - {anchor_target}))

        positive_index = np.random.choice(self.target_to_idxs[positive_target])
        negative_index = np.random.choice(self.target_to_idxs[negative_target])

        sample1 = self.loader(self.samples[index][0])
        sample2 = self.loader(self.samples[positive_index][0])
        sample3 = self.loader(self.samples[negative_index][0])
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample3 = self.transform(sample3)

        return (sample1, sample2, sample3), []


class BalancedBatchSampler(BatchSampler):

    def __init__(self, targets, n_classes, n_samples):
        self.targets = targets
        self.classes = list(set(self.targets))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.targets)
        self.batch_size = self.n_classes * self.n_samples

        self.target_to_idxs = {target: np.where(np.array(self.targets) == target)[0] for target in self.classes}
        np.random.seed(42)

    def __iter__(self):
        count = 0
        while count + self.batch_size < self.n_dataset:
            indices = []
            for target in np.random.choice(self.classes, self.n_classes, replace=False):
                indices.extend(np.random.choice(self.target_to_idxs[target], self.n_samples, replace=False))
            yield indices
            count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
