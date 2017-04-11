import torch
import torch.nn
from src.utils import var
from torch.autograd import Variable


class KLN01Loss(torch.nn.Module):

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), 'wtf?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            # mu_1 = 0; sigma_1 = 1

            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # mu_2 = 0; sigma_2 = 1

            t1 = (samples_var.pow(2) + samples_mean.pow(2)) / 2
            t2 = -samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL


# ----------------------------

def pairwise_euclidean(samples):

    B = samples.size(0)

    samples_norm = samples.mul(samples).sum(1)
    samples_norm = samples_norm.expand(B, B)

    dist_mat = samples.mm(samples.t()).mul(-2) + \
        samples_norm.add(samples_norm.t())
    return dist_mat


def sample_entropy(samples):

        # Assume B x C input

    dist_mat = pairwise_euclidean(samples)

    # Get max and add it to diag
    m = dist_mat.max().detach()
    dist_mat_d = dist_mat + \
        Variable(torch.eye(dist_mat.size(0)) * (m.data[0] + 1)).cuda()

    entropy = (dist_mat_d.min(1)[0] + 1e-4).log().sum()

    entropy *= (samples.size(1) + 0.) / samples.size(0)

    return entropy


class SampleKLN01Loss(torch.nn.Module):

    def __init__(self, direction, minimize):
        super(SampleKLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        

        assert samples.ndimension == 2, 'wft'
        samples = samples.view(samples.size(0), -1)

        self.samples_var = var(samples)
        self.samples_mean = samples.mean(0)

        if self.direction == 'pq':
            assert False, 'not possible'
        else:
            entropy = sample_entropy(samples)

            cross_entropy = - samples.pow(2).mean() / 2.

            KL = - cross_entropy - entropy

        if not self.minimize:
            KL *= -1

        return KL
