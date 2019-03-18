
import os
import sys

from model import video

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

model = build_ssd('test', 300, 21)    # initialize SSD
model.load_weights('../weights/vgg16_reducedfc.pth')
model = model.cuda()

model.eval()
with torch.no_grad():
    for im in video.get_frames():
        im = im.view((-1,) + im.size()).cuda()
        print(im.size())
        output = model.forward(im)
        print(output.size())