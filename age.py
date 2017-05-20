from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import time
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from src.utils import *
import src.losses as losses
from skimage import color

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', type=str, help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=12)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int)

parser.add_argument('--nepoch', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--netG', default='',
                    help="path to netG config")
parser.add_argument('--netE', default='',
                    help="path to netE config")
parser.add_argument('--netG_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netE_chp', default='',
                    help="path to netE (to continue training)")

parser.add_argument('--save_dir', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--match_z', default='cos', help='none|L1|L2|cos')
parser.add_argument('--match_x', default='L1', help='none|L1|L2|cos')


parser.add_argument('--th', default=1.05, type=float, help='')
parser.add_argument('--drop_lr', default=2, type=int, help='')
parser.add_argument('--save_every', default=50, type=int, help='')

parser.add_argument('--adaptive_advesary_weight', default=False,
                    dest='adaptive_advesary_weight',
                    action='store_true', help='manual seed')

parser.add_argument('--feature', dest='feature', action='store_true')
parser.add_argument('--no-feature', dest='feature', action='store_false')

parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch number to start with')

parser.add_argument(
    '--e_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:0",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--g_updates', default="2;KL_fake:1,match_z:1,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)

opt = parser.parse_args()

# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
dataloader = dict(train=setup_dataset(opt, train=True),
                  val=setup_dataset(opt, train=False))

# Load generator
netG = load_G(opt)

# Load encoder
netE = load_E(opt)

x = torch.FloatTensor(opt.batch_size, opt.nc,
                      opt.image_size, opt.image_size)
z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1)
fixed_z = torch.FloatTensor(64, opt.nz, 1, 1).normal_(0, 1)
x_val = torch.FloatTensor(64, opt.nc,
                      opt.image_size, opt.image_size)

if opt.noise == 'sphere':
    normalize_(fixed_z)

if opt.cuda:
    netE.cuda()
    netG.cuda()
    x_val, x = x_val.cuda(), x.cuda()
    z, fixed_z = z.cuda(), fixed_z.cuda()

x, x_val   = Variable(x), Variable(x_val)
z, fixed_z = Variable(z), Variable(fixed_z)

# Setup optimizers
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Setup criterions
if opt.criterion == 'param':
    print('Using parametric criterion KL_%s' % opt.KL)
    KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
    KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
elif opt.criterion == 'nonparam':
    print('Using NON-parametric criterion KL_%s' % opt.KL)
    KL_minimizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=True)
    KL_maximizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=False)
else:
    assert False, 'criterion?'

real_cpu = torch.FloatTensor()


def deprocess(tensor):
    if 'colorization' in opt.dataset:
        out = torch.FloatTensor(*tensor.size())

        for i in range(tensor.size(0)):
            this = tensor[i].transpose(0, 2).contiguous().numpy()

            color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])

            out[i] = torch.FloatTensor(
                color.lab2rgb(this.astype(np.float64))).transpose(0, 2).contiguous()
        return out
    else:
        return tensor / 2 + 0.5


def save_images(epoch):

    real_cpu.resize_(x.data.size()).copy_(x.data)

    # Real samples
    save_path = '%s/real_samples.png' % opt.save_dir
    vutils.save_image(deprocess(real_cpu[:64]), save_path)

    netG.eval()

    populate_x(x_val, dataloader['val'])
    fake = netG(conditional(x_val), fixed_z)

    # Fake samples
    save_path = '%s/fake_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(deprocess(fake.data[:64].cpu()), save_path)

    # Save reconstructions
    gex = netG(conditional(x_val), netE(x_val))

    t = torch.FloatTensor(x_val.size(0) * 2, x_val.size(1),
                          x_val.size(2), x_val.size(3))

    t[0::2] = x_val.data[:]
    t[1::2] = gex.data[:]

    save_path = '%s/reconstructions_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(deprocess(t[:64]), save_path)

    netG.train()

start_lr = opt.lr


def adjust_lr(epoch):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr *= 0.96
        for param_group in optimizerE.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr


def conditional(x):
    if 'colorization' in opt.dataset:
        return x[:, :1, :, :].clone()
    else:
        return None

ideal_KL = get_ideal_KL(opt.nz)

stats = {}
eee = 100000
for epoch in range(opt.start_epoch, opt.nepoch):

    # Adjust learning rate
    adjust_lr(epoch)

    for i in range(len(dataloader['train'])):

        if i == eee:
            start = time.time()
        # ---------------------------
        #        Optimize over e
        # ---------------------------
        for p in netG.parameters():
            p.requires_grad = False
        for p in netE.parameters():
            p.requires_grad = True

        for e_iter in range(updates['e']['num_updates']):
            e_losses = []
            netE.zero_grad()

            # X
            populate_x(x, dataloader['train'])
            # e(X)
            ex = netE(x)

            # KL_real: - \Delta( e(X) , Z ) -> max_e
            KL_real = KL_minimizer(ex)
            e_losses.append(KL_real * updates['e']['KL_real'])

            if updates['e']['match_x'] != 0:
                # g(e(X))
                gex = netG(conditional(x), ex)

                # match_x: E_x||g(e(x)) - x|| -> min_e
                err = match(gex, x, opt.match_x)
                e_losses.append(err * updates['e']['match_x'])

            # Save some stats
            stats['real_mean'] = KL_minimizer.samples_mean.data.mean()
            stats['real_var'] = KL_minimizer.samples_var.data.mean()
            stats['KL_real'] = KL_real.data[0]

            # ================================================

            # Z
            populate_z(z, opt)
            # g(Z)
            fake = netG(conditional(x), z).detach()
            # e(g(Z))
            egz = netE(fake)

            # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
            KL_fake = KL_maximizer(egz)
            e_losses.append(KL_fake * updates['e']['KL_fake'])

            if updates['e']['match_z'] != 0:
                # match_z: E_z||e(g(z)) - z|| -> min_e
                err = match(egz, z, opt.match_z)
                e_losses.append(err * updates['e']['match_z'])

            # Save some stats
            stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
            stats['fake_var'] = KL_maximizer.samples_var.data.mean()
            stats['KL_fake'] = -KL_fake.data[0]

            # Update e
            sum(e_losses).backward()
            optimizerE.step()

        # ---------------------------
        #        Minimize over g
        # ---------------------------
        for p in netE.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netG.parameters():
            p.requires_grad = True

        for g_iter in range(updates['g']['num_updates']):
            g_losses = []
            netG.zero_grad()

            # Z
            populate_z(z, opt)
            populate_x(x, dataloader['train'])
            # g(Z)
            fake = netG(conditional(x), z)
            # e(g(Z))
            egz = netE(fake)

            # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
            KL_fake_g = KL_minimizer(egz)
            g_losses.append(KL_fake_g * updates['g']['KL_fake'])

            if updates['g']['match_z'] != 0:
                # match_z: E_z||e(g(z)) - z|| -> min_g
                err = match(egz, z, opt.match_z)
                err = err * updates['g']['match_z']
                g_losses.append(err)

            # ==================================

            if updates['g']['match_x'] != 0:
                # X
                populate_x(x, dataloader['train'])
                # e(X)
                ex = netE(x)

                # g(e(X))
                gex = netG(conditional(x), ex)

                # match_x: E_x||g(e(x)) - x|| -> min_g
                err = match(gex, x, opt.match_x)
                err = err * updates['g']['match_x']
                g_losses.append(err)

            # Step g
            sum(g_losses).backward()
            optimizerG.step()

        print('[{epoch}/{nepoch}][{iter}/{niter}]  '
              'KL_real/fake: {KL_real:5.3f}/{KL_fake:5.3f}  '
              'mean_real/fake: {real_mean:6.3f}/{fake_mean:6.3f}  '
              'var_real/fake: {real_var:.3f}/{fake_var:.3f}  '
              'adv_weigh {adv_weight}'.format(epoch=epoch,
                                              nepoch=opt.nepoch,
                                              iter=('{:%dd}' % len(
                                                  str(len(dataloader['train']) + 100000))).format(i),
                                              niter=len(dataloader['train']),
                                              adv_weight=updates[
                                                  'e']['KL_fake'],
                                              **stats))
        if i >= eee:
            print((time.time() - start) / (i - eee + 1))

        if opt.adaptive_advesary_weight:
            updates['e']['KL_fake'] += 0.001 * (opt.lr / start_lr)
            if stats['KL_real'] > ideal_KL * opt.th or stats['KL_fake'] > ideal_KL * opt.th:
                updates['e']['KL_fake'] -= 0.02 * (opt.lr / start_lr)
                updates['e']['KL_fake'] = max(0, updates['e']['KL_fake'])

        if i % opt.save_every == 0:
            save_images(epoch)

        # If an epoch takes long time, dump intermediate
        if opt.dataset in ['lsun', 'imagenet'] and (i % 5000 == 0):
            torch.save(netG, '%s/netG_epoch_%d_it_%d.pth' %
                       (opt.save_dir, epoch, i))
            torch.save(netE, '%s/netE_epoch_%d_it_%d.pth' %
                       (opt.save_dir, epoch, i))

    # do checkpointing
    torch.save(netG, '%s/netG_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netE, '%s/netE_epoch_%d.pth' % (opt.save_dir, epoch))
