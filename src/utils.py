import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
import importlib
from .dataset import FolderWithImages
import random
import os
import torch.backends.cudnn as cudnn
from PIL import Image


def setup(opt):
    '''
    Setups cudnn, seeds and parses updates string.
    '''
    opt.cuda = not opt.cpu

    torch.set_num_threads(4)

    if opt.nc is None:
        opt.nc = 1 if opt.dataset == 'mnist' else 3

    try:
        os.makedirs(opt.save_dir)
    except OSError:
        print('Directory was not created.')

    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)

    print("Random Seed: ", opt.manual_seed)
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device,"
              "so you should probably run with --cuda")

    updates = {'e': {}, 'g': {}}
    updates['e']['num_updates'] = int(opt.e_updates.split(';')[0])
    updates['e'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.e_updates.split(';')[1].split(',')})

    updates['g']['num_updates'] = int(opt.g_updates.split(';')[0])
    updates['g'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.g_updates.split(';')[1].split(',')})

    return updates


def setup_dataset(opt, train=True, shuffle=True, drop_last=True):
    '''
    Setups dataset.
    '''
    # Usual transform
    t = transforms.Compose([
        transforms.Scale([opt.image_size, opt.image_size]),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        imdir = 'train' if train else 'val'
        dataroot = os.path.join(opt.dataroot, imdir)

        dataset = dset.ImageFolder(root=dataroot, transform=t)
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot,
                            classes=['bedroom_train'],
                            train=train,
                            transform=t)
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root='data/raw/cifar10',
                               download=True,
                               train=train,
                               transform=t
                               )
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root='data/raw/mnist',
                             download=True,
                             train=train,
                             transform=t
                             )
    elif opt.dataset == 'svhn':
        dataset = dset.SVHN(root='data/raw/svhn',
                            download=True,
                            train=train,
                            transform=t)
    elif opt.dataset == 'celeba':
        imdir = 'train' if train else 'val'
        dataroot = os.path.join(opt.dataroot, imdir)

        dataset = FolderWithImages(root=dataroot,
                                   input_transform=transforms.Compose([
                                       ALICropAndScale(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]),
                                   target_transform=transforms.ToTensor()
                                   )

    else:
        assert False, 'Wrong dataset name.'

    assert len(dataset) > 0, 'No images found, check your paths.'

    # Shuffle and drop last when training
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=int(opt.workers),
                                             pin_memory=True,
                                             drop_last=drop_last)

    return InfiniteDataLoader(dataloader)


class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)


def weights_init(m):
    '''
    Custom weights initialization called on netG and netE
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_G(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netG)
    netG = m._netG(opt)
    netG.apply(weights_init)
    netG.train()
    if opt.netG_chp != '':
        netG.load_state_dict(torch.load(opt.netG_chp).state_dict())

    print('Generator\n', netG)
    return netG


def load_E(opt):
    '''
    Loads encoder model.
    '''
    m = importlib.import_module('models.' + opt.netE)
    netE = m._netE(opt)
    netE.apply(weights_init)
    netE.train()
    if opt.netE_chp != '':
        netE.load_state_dict(torch.load(opt.netE_chp).state_dict())

    print('Encoder\n', netE)

    return netE


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)

        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


def populate_x(x, dataloader):
    '''
    Fills input variable `x` with data generated with dataloader
    '''
    real_cpu, _ = dataloader.next()
    x.data.resize_(real_cpu.size()).copy_(real_cpu)


def populate_z(z, opt):
    '''
    Fills noise variable `z` with noise U(S^M)
    '''
    z.data.resize_(opt.batch_size, opt.nz, 1, 1)
    z.data.normal_(0, 1)
    if opt.noise == 'sphere':
        normalize_(z.data)


def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    x.div_(x.norm(2, dim=dim).expand_as(x))


def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).expand_as(x))


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
