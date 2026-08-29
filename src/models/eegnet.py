import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.5):
        super(EEGNet, self).__init__()
        
        # 1. Temporal Convolution: Learns frequency filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        # 2. Depthwise Convolution: Learns spatial filters (The "Topomaps")
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # 3. Separable Convolution: Combines temporal and spatial info
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # 4. Classifier
        self.classifier = nn.Linear(32 * (Samples // 32), nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
