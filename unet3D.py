import torch
import torch.nn as nn
from torch.nn import functional as F

# 每一层上的两次卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels // 2    # 注意通道数要为Int，需要使用//进行除法(得到int。/得到float)
        if in_channels > out_channels:
            channels = in_channels // 2

        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]

        if batch_normal:  # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造同一层卷积的序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=False):
        super(UpSampling, self).__init__()
        if bilinear:    # 双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:   # 反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels, batch_normal)

    # input1是上采样特征图，input2是下采样传来的特征图
    def forward(self, input1, input2):
        input1 = self.up(input1)
        # 特征图融合
        output = torch.cat([input2, input1], dim=1)
        output = self.conv(output)
        return output

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):

    # 这边bilinear为Ture的时候会报错。

    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=False):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        self.inputs = DoubleConv(in_channels, 64, batch_normal=self.batch_normal)
        self.down_1 = DownSampling(64, 128, batch_normal=self.batch_normal)
        self.down_2 = DownSampling(128, 256, batch_normal=self.batch_normal)
        self.down_3 = DownSampling(256, 512, batch_normal=self.batch_normal)

        self.up_1 = UpSampling(512, 256, bilinear=self.bilinear)
        self.up_2 = UpSampling(256, 128, bilinear=self.bilinear)
        self.up_3 = UpSampling(128, 64, batch_normal=self.batch_normal)
        self.output = LastConv(64, num_classes)

    def forward(self, x):
        # 下采样
        x1 = self.inputs(x)     # [2, 64, 72, 88, 104]
        x2 = self.down_1(x1)    # [2, 128, 36, 44, 52]
        x3 = self.down_2(x2)    # [2, 256, 18, 22, 26]
        x4 = self.down_3(x3)    # [2, 512, 9, 11, 13]

        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.output(x7)
        return x