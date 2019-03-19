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
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    kalman = KalmanTracking()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
    gt = read_annotations('../datasets/AICity_data/train/S03/c010/m6-full_annotation.xml')
    dataset = YoloDataset(video, gt, detection_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for epoch in tqdm(range(10), file=sys.stdout, desc='Fine tuning...'):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(imgs, targets)
            loss.backward()
            optimizer.step()
