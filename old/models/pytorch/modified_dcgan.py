from __future__ import print_function
import argparse
import numpy as np
import os
import pdb
import random
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from load_data import truly_random_dataset, rescale, to_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=6854 ,help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


dataset = truly_random_dataset(opt.dataroot, opt.imageSize, transform = transforms.Compose([rescale(), to_tensor()]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


def bad_prng(shapes):
    # IBM's random generator: RANDU
    num_iters = np.prod(shapes)
    seq = []
    new_num = opt.manualSeed
    for i in range(int(num_iters)):
        new_num = (65539*new_num) % (2**31)
        seq.append(new_num)

    a = np.log(np.asarray(seq))
    new_seq = a*-1 + 2*np.max(a)
    new_seq -= np.mean(new_seq)
    new_seq /= np.max(new_seq)

    b_noise = new_seq.reshape(shapes)
    opt.manualSeed = new_num
    return b_noise


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print('shape: ',m.weight.data.shape)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        print('shape 2: ',m.weight.data.shape)

count = 0
coef = 8
coef_elif = 8
def weights_init_non_random_G(m):
    global count
    global coef, coef_elif
    global nz
    global ngf
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        dtype = torch.FloatTensor
        if count == 0:
            shapes = np.array([nz, ngf*coef, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        elif count == 4:
            shapes = np.array([ngf*coef*2, 1, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        else:
            shapes = np.array([ngf*coef*2, ngf*coef, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        coef *= 0.5
        count += 1

    elif classname.find('BatchNorm') != -1:
        shapes = np.array([ngf*coef_elif])
        m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        m.bias.data.fill_(0)
        coef_elif *= 0.5


countg = 0
coefg = 1
coefg_elif = 2        
def weights_init_non_random_D(m):
    global countg
    global coefg, coefg_elif
    global nc
    global ndf
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        dtype = torch.FloatTensor
        if countg == 0:
            shapes = np.array([ndf, nc, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        elif countg == 4:
            shapes = np.array([1,ndf*coefg/2, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        else:
            shapes = np.array([ndf*coefg , ndf*coefg/2, 4, 4 ])
            m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        coefg *= 2
        countg += 1

    elif classname.find('BatchNorm') != -1:
        shapes = np.array([ndf*coefg_elif])
        m.weight.data = torch.FloatTensor(bad_prng(shapes.astype(int)))
        m.bias.data.fill_(0)
        coefg_elif *= 2
 
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # torch.nn.ConvTranspose2d(in_channels, out_channels, 
            # kernel_size, stride=1, padding=0, output_padding=0, 
            # groups=1, bias=True, dilation=1)
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init_non_random_G)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # torch.nn.Conv2d(in_channels, out_channels, 
            # kernel_size, stride=1, padding=0, dilation=1, 
            # groups=1, bias=True)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init_non_random_D)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

###
#fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)


shapes = np.array([opt.batchSize, nz, 1, 1])
fixed_noise = torch.FloatTensor(bad_prng(shapes))
#fixed_noise = torch.FloatTensor(((inv_cdf-np.min(inv_cdf))/(2*np.max(inv_cdf))).reshape(shapes.astype(int)))

###

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    #noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_noise = fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        for rep in range(1):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real

            netD.zero_grad()
            real_cpu = data
            #pdb.set_trace()
            
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = netD(inputv)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            #noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            
            shapes = np.array([batch_size, nz, 1, 1])
            noise = torch.FloatTensor(bad_prng(shapes))
            ###
            if opt.cuda:
                noise = = noise.cuda()

            noisev = Variable(noise)
            fake = netG(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = netD(fake.detach())
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            fake.data[fake.data <= 0] = 0
            fake.data[fake.data > 0] = 1
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

            _data = np.array(fake.data.cpu().numpy().reshape((batch_size, -1)), dtype = int)
            for _batch in range(batch_size):
                _f = open('%s/fake_samples_epoch_%03d_batch_%d.txt' % (opt.outf, epoch, _batch), 'w')
                _f.writelines("\n".join([str(_) for _ in _data[_batch]]))
                _f.close()

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))