import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ChannelSELayer


class Prior(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.d1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.d2 = nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2, groups=in_ch, bias=False)
        self.d3 = nn.Conv2d(in_ch, in_ch, 3, padding=4, dilation=4, groups=in_ch, bias=False)
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1, bias=False)
        )

    def forward(self, x):
        g1 = self.d1(x).pow(2)
        g2 = self.d2(x).pow(2)
        g3 = self.d3(x).pow(2)
        v = self.pw(torch.cat([g1, g2, g3], dim=1))
        return torch.tanh(v)


class InterpBlock(nn.Module):
    def __init__(self, scale=2, alpha=1.6):
        super().__init__()
        self.scale = scale
        self.alpha = alpha

    def forward(self, x, vessel_prior):
        x_up = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        v_up = F.interpolate(vessel_prior, scale_factor=self.scale, mode='bilinear', align_corners=False)
        output = x_up + self.alpha * x_up * v_up
        return output, v_up


class RollerCoaster(nn.Module):
    def __init__(self, in_ch, feat_ch=32, scale=2):
        super().__init__()
        self.prior = Prior(in_ch)
        self.upsample = InterpBlock(scale=scale, alpha=1.5)
        self.se = ChannelSELayer(spatial_dims=2, in_channels=in_ch + 1, r=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + 1, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        v = self.prior(x)
        x, v_up = self.upsample(x, v)
        x = torch.cat([x, v_up], dim=1)
        x = self.se(x)
        x = self.fuse(x)
        return x, v_up


class MultiScaleDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        f0 = self.conv(x)
        x1 = self.pool(x)
        f1 = self.conv(x1)
        f1 = F.interpolate(f1, size=f0.shape[2:], mode='bilinear', align_corners=False)
        x2 = self.pool(x1)
        f2 = self.conv(x2)
        f2 = F.interpolate(f2, size=f0.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([f0, f1, f2], dim=1)
        return self.fuse(out)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        if use_attention:
            self.attn = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        if self.use_attention:
            x2 = self.attn(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DeepSupervisionAttentionUnet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 4, base_ch * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 8, base_ch * 16))

        self.up1 = Up(base_ch * 16, base_ch * 8, use_attention=True)
        self.up2 = Up(base_ch * 8, base_ch * 4, use_attention=True)
        self.up3 = Up(base_ch * 4, base_ch * 2, use_attention=True)

        self.aux_head = nn.Conv2d(base_ch * 2, out_ch, kernel_size=1)
        self.up4 = Up(base_ch * 2, base_ch, use_attention=True)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        aux_logits = self.aux_head(x)

        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if aux_logits.shape[-2:] != x.shape[-2:]:
            aux_logits = F.interpolate(aux_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return logits, aux_logits


class RollerCoasterSNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feat_ch=32, scale=2):
        super().__init__()
        self.rc = RollerCoaster(in_ch, feat_ch, scale)
        self.ms = MultiScaleDown(feat_ch, feat_ch)
        self.se = ChannelSELayer(spatial_dims=2, in_channels=feat_ch + 1, r=4)
        self.unet = DeepSupervisionAttentionUnet(feat_ch + 1, out_ch, base_ch=32)

    def forward(self, x):
        feat, prior = self.rc(x)
        feat_ms = self.ms(feat)
        unet_input = torch.cat([feat_ms, prior], dim=1)
        unet_input = self.se(unet_input)
        logits, aux_logits = self.unet(unet_input)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if aux_logits.shape[-2:] != x.shape[-2:]:
            aux_logits = F.interpolate(aux_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return logits, aux_logits


def get_model(model_name, device):
    if model_name == "rcnet":
        return RollerCoasterSNet(in_ch=1, out_ch=1).to(device)
    else:
        raise ValueError("Unknown model name")