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

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)
