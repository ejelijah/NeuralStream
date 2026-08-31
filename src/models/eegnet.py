import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.6): # Increased Dropout to 0.6
        super(EEGNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)
            dummy_out = self.block3(self.block2(self.block1(dummy_input)))
            self.flatten_size = dummy_out.view(1, -1).size(1)
        
        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


