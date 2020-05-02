import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, inputs, outputs, stride=1, downsample=None, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.conv3 = nn.Conv2d(outputs, outputs * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(outputs * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
