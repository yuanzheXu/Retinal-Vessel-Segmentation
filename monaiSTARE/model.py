'''E1: Simple CNN U-Net baseline. 3-down 3-up, NO stage0. First layer is downsampling.'''
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Double conv: 3x3 -> BN -> ReLU -> 3x3 -> BN -> ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class DownBlock(nn.Module):
    """MaxPool(2) + DoubleConv"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)
    def forward(self, x): return self.conv(self.pool(x))

class UpBlock(nn.Module):
    """TransposeConv(2x) + concat(skip) + DoubleConv"""
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.conv = ConvBlock(out_c + skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class SimpleUNet(nn.Module):
    """3-level U-Net: each encoder level starts with downsampling. No stage0."""
    def __init__(self, in_channels=1, num_classes=1, base_c=64):
        super().__init__()
        C = base_c
        # Encoder: 3 levels, all downsampling-first
        self.enc1 = DownBlock(in_channels, C)     # H -> H/2
        self.enc2 = DownBlock(C, C * 2)            # H/2 -> H/4
        self.bottleneck = DownBlock(C * 2, C * 4)  # H/4 -> H/8

        # Decoder: 3 levels, upsample + concat skip
        self.dec2 = UpBlock(C * 4, C * 2, C * 2)   # H/8 -> H/4, skip from enc2
        self.dec1 = UpBlock(C * 2, C, C)            # H/4 -> H/2, skip from enc1
        self.dec0 = UpBlock(C, C, in_channels)      # H/2 -> H, skip from input

        self.out = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                    # H/2
        e2 = self.enc2(e1)                   # H/4
        b = self.bottleneck(e2)              # H/8
        d2 = self.dec2(b, e2)                # H/4
        d1 = self.dec1(d2, e1)               # H/2
        d0 = self.dec0(d1, x)                # H
        return self.out(d0)

def get_model(model_name=None, device="cuda", base_c=None):
    if base_c is None: base_c = 64
    return SimpleUNet(in_channels=1, num_classes=1, base_c=base_c).to(device)
