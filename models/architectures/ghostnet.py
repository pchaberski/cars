"""
Based on implementation by iamhankai:
https://github.com/iamhankai/ghostnet.pytorch/blob/master/ghost_net.py

GhostNet paper: <https://arxiv.org/abs/1911.11907>
"""


import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),)

    def forward(self, input):
        b, c, _, _ = input.size()
        output = self.avg_pool(input).view(b, c)
        output = self.fc(output).view(b, c, 1, 1)
        output = torch.clamp(output, 0, 1)
        output = output * input
        return output


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, input):
        output_1 = self.primary_conv(input)
        output_2 = self.cheap_operation(output_1)
        output = torch.cat([output_1, output_2], dim=1)
        return output[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super().__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # Point-wise expansion
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # Depth-wise convolution
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-excitation
            SqueezeExcite(hidden_dim) if use_se else nn.Sequential(),
            # Point-wise linear projection
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, input):
        output = self.conv(input) + self.shortcut(input)
        return output


class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, img_channels=3, dropout=0.2, out_channels=1280, width_mult=1.):
        super().__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.dropout = dropout
        self.out_channels = out_channels

        # setting of inverted residual blocks
        self.cfgs = [
            # k, t, c, SE, s
            [3,  16,  16, 0, 1],
            [3,  48,  24, 0, 2],
            [3,  72,  24, 0, 1],
            [5,  72,  40, 1, 2],
            [5, 120,  40, 1, 1],
            [3, 240,  80, 0, 2],
            [3, 200,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(self.img_channels, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = self.out_channels
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(output_channel, self.num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        output = self.features(input)
        output = self.squeeze(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
