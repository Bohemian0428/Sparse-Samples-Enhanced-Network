import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """SegNet (Badrinarayanan et al., 2017).

    基于 VGG16 结构的编码器-解码器，编码器保存最大池化索引，
    解码器通过 max-unpooling 恢复空间分辨率，最后接 1x1 卷积 + softmax。
    """

    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        def conv_block(in_c, out_c, n):
            layers = []
            for i in range(n):
                layers.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # 编码器（VGG16：2-2-3-3-3）
        self.enc1 = conv_block(in_channels, 64, 2)
        self.enc2 = conv_block(64, 128, 2)
        self.enc3 = conv_block(128, 256, 3)
        self.enc4 = conv_block(256, 512, 3)
        self.enc5 = conv_block(512, 512, 3)

        # 解码器（镜像，通道逐级减半）
        self.dec5 = conv_block(512, 512, 3)
        self.dec4 = conv_block(512, 256, 3)
        self.dec3 = conv_block(256, 128, 3)
        self.dec2 = conv_block(128, 64, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1),
        )

    def forward(self, x):
        # 编码
        x = self.enc1(x)
        x, i1 = self.pool(x)
        x = self.enc2(x)
        x, i2 = self.pool(x)
        x = self.enc3(x)
        x, i3 = self.pool(x)
        x = self.enc4(x)
        x, i4 = self.pool(x)
        x = self.enc5(x)
        x, i5 = self.pool(x)

        # 解码
        x = F.max_unpool2d(x, i5, kernel_size=2, stride=2)
        x = self.dec5(x)
        x = F.max_unpool2d(x, i4, kernel_size=2, stride=2)
        x = self.dec4(x)
        x = F.max_unpool2d(x, i3, kernel_size=2, stride=2)
        x = self.dec3(x)
        x = F.max_unpool2d(x, i2, kernel_size=2, stride=2)
        x = self.dec2(x)
        x = F.max_unpool2d(x, i1, kernel_size=2, stride=2)
        x = self.dec1(x)

        # 分类：softmax；回归（num_classes==1）：输出原始高度，不加 softmax
        if self.num_classes > 1:
            x = F.softmax(x, dim=1)
        return x


def get_segnet(in_channels, num_classes):
    return SegNet(in_channels, num_classes)


if __name__ == '__main__':
    m = get_segnet(13, 6)
    x = torch.randn(2, 13, 256, 256)
    y = m(x)
    print('SegNet output:', y.shape)
    print('params (M): %.2f' % (sum(p.numel() for p in m.parameters()) / 1e6))
