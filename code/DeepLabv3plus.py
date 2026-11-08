import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet-50 编码器，输出低层特征(OS=4)与高层特征(OS=16)。"""

    def __init__(self, in_channels=13, layers=(3, 4, 6, 3), output_stride=16):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilation=1)

        if output_stride == 16:
            self.layer3 = self._make_layer(256, layers[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=2)
        elif output_stride == 8:
            self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4)
        else:
            raise ValueError('output_stride must be 8 or 16')

    def _make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.in_planes, planes, stride, dilation, downsample)]
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, 1, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # OS=2
        x = self.maxpool(x)                       # OS=4
        x = self.layer1(x)                        # OS=4, 256 通道
        low = x
        x = self.layer2(x)                        # OS=8
        x = self.layer3(x)                        # OS=16
        x = self.layer4(x)                        # OS=16, 2048 通道
        return low, x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling，多尺度空洞卷积 + 全局平均池化分支。"""

    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.rate_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for r in atrous_rates
        ])
        # 注意：全局池化分支输出为 1x1 特征，去掉 BatchNorm 以避免 batch=1 时
        # "Expected more than 1 value per channel" 报错（小数据集最后一个 batch 常为 1）。
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        outs = [self.conv1x1(x)]
        for conv in self.rate_convs:
            outs.append(conv(x))
        g = self.global_pool(x)
        g = F.interpolate(g, size=size, mode='bilinear', align_corners=False)
        outs.append(g)
        x = torch.cat(outs, dim=1)
        return self.project(x)


class DeepLabv3Plus(nn.Module):
    """DeepLabv3+ (Chen et al., 2018)：ResNet-50 骨干 + ASPP + 解码器。"""

    def __init__(self, in_channels, num_classes, output_stride=16):
        super(DeepLabv3Plus, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNetEncoder(in_channels, output_stride=output_stride)
        self.aspp = ASPP(2048, 256)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(48 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low, high = self.backbone(x)           # low: OS=4, high: OS=16
        high = self.aspp(high)                  # OS=16, 256 通道
        high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        low = self.low_level_project(low)       # 48 通道
        x = torch.cat([high, low], dim=1)       # 256 + 48
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        # 分类：softmax；回归（num_classes==1）：输出原始高度，不加 softmax
        if self.num_classes > 1:
            x = F.softmax(x, dim=1)
        return x


def get_deeplabv3plus(in_channels, num_classes, output_stride=16):
    return DeepLabv3Plus(in_channels, num_classes, output_stride=output_stride)


if __name__ == '__main__':
    m = get_deeplabv3plus(13, 6)
    x = torch.randn(2, 13, 256, 256)
    y = m(x)
    print('DeepLabv3+ output:', y.shape)
    print('params (M): %.2f' % (sum(p.numel() for p in m.parameters()) / 1e6))
