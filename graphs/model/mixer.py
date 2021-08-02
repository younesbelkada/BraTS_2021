import torch.nn as nn
import torch

class Encoder_2Dblock(nn.Module):
    def __init__(self):
        super().__init__()

        #self.norm = nn.BatchNorm3d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=500, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
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
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #exit(0)
        out = self.fc(x)
        return out

class MLP_Mixer(nn.Module):
    def __init__(self):
        super().__init__()

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
        self.mixers = nn.ModuleList([Encoder_2Dblock().cuda() for i in range(self.nb_mixers)])


    def forward(self, x):
        embeddings = [self.mixers[i](x[:, i, :, :, :]) for i in range(self.nb_mixers)]
        embeddings = torch.cat(embeddings).view(x.size(0), -1)
        out = self.mixer_net(embeddings)
        return out
