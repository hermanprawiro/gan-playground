import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import bigan_128
from utils.criterion import GANLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--ndf", type=int, default=64, help="Base features multiplier for discriminator")
parser.add_argument("--ngf", type=int, default=64, help="Base features multiplier for generator")
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="bigan")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.save_name)
    args.result_path = os.path.join(args.result_path, args.save_name)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    tfs = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = torchvision.datasets.CelebA(args.data_root, split="all", transform=tfs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netG = bigan_128.Generator(latent_dim=args.latent_dim, ngf=args.ngf, img_dim=args.image_ch).to(device)
    netD = bigan_128.Discriminator(latent_dim=args.latent_dim, ndf=args.ndf, img_dim=args.image_ch).to(device)
    netE = bigan_128.Encoder(latent_dim=args.latent_dim, ndf=args.ndf, img_dim=args.image_ch).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netE.apply(weights_init)

    optG = torch.optim.Adam(list(netG.parameters()) + list(netE.parameters()), lr=2e-4, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(args.beta1, args.beta2))

    criterion = GANLoss('vanilla', target_real_label=0.9, target_fake_label=0.1, target_fake_G_label=0.9).to(device)

    fixed_noise = torch.randn(32, args.latent_dim, device=device)

    netG.train()
    netD.train()
    netE.train()
    for epoch in range(args.n_epochs):
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)

            z_real = torch.randn(inputs.shape[0], args.latent_dim, device=device)
            z_enc = netE(inputs)
            x_fake = netG(z_real)

            optG.zero_grad()
            # Encoder
            outD = netD(inputs, z_enc)
            Dx2 = outD.mean().item()
            lossE = criterion(outD, False)
            lossE.backward()

            # Generator
            outD = netD(x_fake, z_real)
            Dgz2 = outD.mean().item()
            lossG = criterion(outD, False, True)
            lossG.backward()
            optG.step()

            optD.zero_grad()
            # Discriminator (Real)
            outD = netD(inputs, z_enc.detach())
            Dx1 = outD.mean().item()
            lossD_real = criterion(outD, True)
            lossD_real.backward()

            # Discriminator (Fake)
            outD = netD(x_fake.detach(), z_real)
            Dgz1 = outD.mean().item()
            lossD_fake = criterion(outD, False)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optD.step()

            # optG.zero_grad()
            # # Encoder
            # outD = netD(inputs, z_enc)
            # Dx2 = outD.mean().item()
            # lossE = criterion(outD, False)
            # lossE.backward()

            # # Generator
            # outD = netD(x_fake, z_real)
            # Dgz2 = outD.mean().item()
            # lossG = criterion(outD, False, True)
            # lossG.backward()
            # optG.step()

            if i % 50 == 0:
                outG = netG(fixed_noise).detach()
                z_enc = netE(outG).detach()
                reconG = netG(z_enc).detach()
                eval_dist = (fixed_noise - z_enc).norm()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_E: %.4f D(x): %.4f / %.4f D(G(z)): %.4f / %.4f | Dist: %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), lossD.item(), lossG.item(), lossE.item(), Dx1, Dx2, Dgz1, Dgz2, eval_dist))
                save_image(torch.cat([outG, reconG], dim=0), '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1))

        save_model((netG, netD, netE), (optG, optD), epoch, args.checkpoint_path)


def save_model(models, optimizers, epoch, checkpoint_path):
    netG, netD, netE = models
    optG, optD = optimizers

    checkpoint = {
        'state_dict': {
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'encoder': netE.state_dict(),
        },
        'optimizer': {
            'generator': optG.state_dict(),
            'discriminator': optD.state_dict(),
            # 'encoder': optE.state_dict(),
        },
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_%03d.pth' % (checkpoint_path, epoch + 1))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
