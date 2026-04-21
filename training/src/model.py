import torch.nn as nn

#custom head
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )