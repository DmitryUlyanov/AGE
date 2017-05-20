import torch
import torch.nn as nn
import src.utils as utils


class _netE_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netE_Base, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        output = output.view(output.size(0), -1)
        if self.noise == 'sphere':
            output = utils.normalize(output)

        return output


class _netG_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netG_Base, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, _, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)

class _netD(nn.Module):
    def __init__(self, opt, main):
        super(_netD, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        output = output.view(output.size(0), -1)
        if self.noise == 'sphere':
            output = utils.normalize(output)

        return output


class _netG(nn.Module):
    def __init__(self, opt, main):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, _, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)

class _netG_cond_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netG_cond_Base, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input, noise):
        # Check noise is either (B,C,1,1) or (B,C)

        assert input.ndimension() == 4, 'input should be 4 dim'

        if noise is None:
            input_ = input
        else:
            assert noise.nelement() == noise.size(0) * noise.size(1), 'wtf'

            sz = input.size()
            noise_ = noise.view(noise.size(0), noise.size(1), 1, 1).expand(
                noise.size(0), noise.size(1), sz[2], sz[3]).contiguous()

            input_ = torch.cat([input, noise_], 1)

        gpu_ids = None

        AB = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            AB = nn.parallel.data_parallel(self.main, input_, gpu_ids)
        else:
            AB = self.main(input_)

        # print(AB.min().data[0], AB.max().data[0], AB.mean().data[0])
        return torch.cat([input, AB], 1)
