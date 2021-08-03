import torch.nn as nn
import torch
from graphs.weights_initializer import weights_init

class Encoder_2Dblock(nn.Module):
    def __init__(self, in_channels=500):
        super().__init__()

        #self.norm = nn.BatchNorm3d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class MLP_Mixer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        #self.norm = nn.BatchNorm3d(1)
        self.nb_mixers = 9
        self.emb_size = 128
        self.mixer_net = nn.Sequential(
            nn.Linear(9*self.emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.mixers = nn.ModuleList([Encoder_2Dblock(self.config.in_channels) for i in range(self.nb_mixers)])
        self.apply(weights_init)

    def forward(self, x):
        embeddings = [self.mixers[i](x[:, :, i, :, :]) for i in range(self.nb_mixers)]
        embeddings = torch.cat(embeddings).view(x.size(0), -1)
        out = self.mixer_net(embeddings)
        return out
