import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init


class ResidualSequential(nn.Sequential):
    def __init__(self, *args):
        super(ResidualSequential, self).__init__(*args)

    def forward(self, x):
        out = super(ResidualSequential, self).forward(x)
        return out + x


def get_residual_block(num_channels):
    layers = [
        nn.ReflectionPad2d(1),
        nn.Conv2d(num_channels, num_channels, 3, 1, padding=0, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(num_channels, num_channels, 3, 1, padding=0, bias=False),
        nn.BatchNorm2d(num_channels),
    ]
    return ResidualSequential(*layers)


class ResNet(nn.Sequential):
    def __init__(self, num_input_channels, num_output_channels, num_blocks, num_channels, upscale_factor):

        layers = [
            nn.Conv2d(num_input_channels, num_channels,
                      kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels,
                      kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        ]

        for i in range(num_blocks):
            layers += [get_residual_block(num_channels)]

        layers += [
            nn.Conv2d(num_channels, num_output_channels *
                      upscale_factor ** 2, (3, 3), (1, 1), (1, 1)),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.PixelShuffle(upscale_factor),
            # nn.Tanh()
        ]

        # init
        conv_layers = filter(lambda x: 'Conv2d' in str(type(x)), layers)
        for l in conv_layers:
            nn.init.orthogonal(l.weight, sqrt(2))

        super(ResNet, self).__init__(*layers)
