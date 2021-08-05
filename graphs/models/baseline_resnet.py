import torch.nn as nn
import torch
from graphs.weights_initializer import weights_init
import torchvision.models as models

class Baseline_Resnet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        pretrained = self.config.pretrained
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

