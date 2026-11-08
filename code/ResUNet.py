# -*- coding: utf-8 -*-
"""ResU-Net：把 U-Net 的双层卷积块替换为残差块（ResBlock）的对照模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """两层 3x3 卷积 + BN + ReLU，带捷径分支（通道或尺寸变化时用 1x1 投影）。"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.short = None

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResUNet(nn.Module):
    def __init__(self, in_channels=13, num_classes=6, ch=(32, 64, 128, 256), mid=512):
        super().__init__()
        c1, c2, c3, c4 = ch

        self.enc1 = ResBlock(in_channels, c1)          # 256 -> 256
        self.enc2 = ResBlock(c1, c2)                   # 128 -> 128
        self.enc3 = ResBlock(c2, c3)                   # 64  -> 64
        self.enc4 = ResBlock(c3, c4)                   # 32  -> 32
        self.mid = ResBlock(c4, mid)                   # 16  -> 16

        self.pool = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(mid, c4, 2, stride=2)
        self.dec4 = ResBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ResBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ResBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ResBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.mid(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # 与项目其余模型（U-Net / SegNet / DeepLabv3+）保持一致，输出 softmax 概率。
        # 训练用的 WeightedCE 直接对输出取 log，若返回 logits 会得到负损失并发散。
        return F.softmax(self.head(d1), dim=1)


def get_resunet(in_channels, num_classes):
    return ResUNet(in_channels, num_classes)


if __name__ == '__main__':
    m = get_resunet(13, 6)
    y = m(torch.randn(2, 13, 256, 256))
    print('ResUNet output:', y.shape)
    print('params (M): %.3f' % (sum(p.numel() for p in m.parameters()) / 1e6))
