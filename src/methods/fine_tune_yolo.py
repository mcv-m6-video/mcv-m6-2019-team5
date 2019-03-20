import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from tqdm import tqdm

from model import Video
from nn import DetectionTransform
from nn.yolo.models import Darknet, parse_model_config
from nn.yolo.utils import utils
from nn.yolo_dataset import YoloDataset
from operations import KalmanTracking
from utils import read_annotations
import matplotlib.pyplot as plt
import numpy as np


def fine_tune_yolo(debug=False):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    detection_transform = DetectionTransform()
    classes = utils.load_classes('../config/coco.names')

    hyperparams = parse_model_config('../config/yolov3.cfg')[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    model = Darknet('../config/yolov3.cfg')
    model.load_weights('../weights/yolov3.weights')
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    kalman = KalmanTracking()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
    gt = read_annotations('../datasets/AICity_data/train/S03/c010/m6-full_annotation.xml')
    dataset = YoloDataset(video, gt, classes, transforms=detection_transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    for epoch in tqdm(range(10), file=sys.stdout, desc='Fine tuning'):
        for images, targets in tqdm(data_loader, file=sys.stdout, desc='Running epoch'):
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            loss = model(images, targets)
            loss.backward()
            optimizer.step()

    print('Training finished. Saving weights...')
    model.save_weights('../weights/fine_tuned_yolo.weights')
    print('Saved weights')
