# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import torch.nn as nn
from .base import _netE_Base, _netG_Base, _netG_cond_Base
from .unet import UNet128
# ------------------------
#         E
# ------------------------


def _netE(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        #
        nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 16 x 16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=True),
    )

    return _netE_Base(opt, main)

# ------------------------
#         G
# ------------------------


def _netG(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz

    main = UNet128(num_input_channels=nz + 1,
                   num_output_channels=2,
                   num_channels=ngf)

    return _netG_cond_Base(opt, main)
