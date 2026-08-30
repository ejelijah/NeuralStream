import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=1000, dropoutRate=0.5):
        super(EEGNet, self).__init__()
        
        # 1. Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        
        # 2. Depthwise Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # 3. Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # --- DYNAMIC SIZE CALCULATION ---
        # We pass a dummy input through the blocks to see the output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)
            dummy_out = self.block3(self.block2(self.block1(dummy_input)))
            self.flatten_size = dummy_out.view(1, -1).size(1)
        
        # 4. Classifier using the calculated flatten_size
        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1) # This flattens the output to 1056
        x = self.classifier(x)    # Now this matches!
        return x

