import torch.nn as nn
from .base import _netE_Base, _netG_Base
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
        # state size. (ndf) x 16 x 16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        nn.Conv2d(ndf * 4, nz, 4, 2, 1, bias=True),
        nn.AvgPool2d(2),
        # nn.BatchNorm2d(nout),

    )

    return _netE_Base(opt, main)

# ------------------------
#         G
# ------------------------


def _netG(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.Conv2d(ngf * 2, nc, 1, bias=True),
        nn.Tanh()
    )

    return _netG_Base(opt, main)
