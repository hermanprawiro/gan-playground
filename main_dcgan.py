import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import dcgan, dcgan_specnorm
from utils.criterion import GANLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--ndf", type=int, default=64, help="Base features multiplier for discriminator")
parser.add_argument("--ngf", type=int, default=64, help="Base features multiplier for generator")
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=64)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="dcgan")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")

def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.save_name)
    args.result_path = os.path.join(args.result_path, args.save_name)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    tfs = transforms.Compose([
        transforms.Resize(args.image_res),
        transforms.CenterCrop(args.image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args.image_ch, [0.5]*args.image_ch)
    ])

    dataset = torchvision.datasets.CelebA(args.data_root, split="all", transform=tfs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netG = dcgan.Generator(z_dim=args.latent_dim, ngf=args.ngf, img_dim=args.image_ch, resolution=args.image_res).to(device)
    netD = dcgan.Discriminator(ndf=args.ndf, img_dim=args.image_ch, resolution=args.image_res).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=4e-4, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    criterion = GANLoss('vanilla', target_real_label=0.9, target_fake_label=0.0, target_fake_G_label=0.9).to(device)

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    netG.train()
    netD.train()
    for epoch in range(args.n_epochs):
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)

            optD.zero_grad()
            # Discriminator (Real)
            outD = netD(inputs)
            Dx = outD.mean().item()
            lossD_real = criterion(outD, True)
            lossD_real.backward()

            # Generate Fake
            z = torch.randn(inputs.shape[0], args.latent_dim, device=device)
            outG = netG(z)

            # Discriminator (Fake)
            outD = netD(outG.detach())
            Dgz1 = outD.mean().item()
            lossD_fake = criterion(outD, False)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optD.step()

            optG.zero_grad()
            # Generator
            outD = netD(outG)
            Dgz2 = outD.mean().item()
            lossG = criterion(outD, False, True)
            lossG.backward()
            optG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), lossD.item(), lossG.item(), Dx, Dgz1, Dgz2))

            if i % 50 == 0:
                outG = netG(fixed_noise).detach()
                save_image(outG, '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1))
        save_model((netG, netD), (optG, optD), epoch, args.checkpoint_path)            


def save_model(models, optimizers, epoch, checkpoint_path):
    netG, netD = models
    optG, optD = optimizers

    checkpoint = {
        'state_dict': {
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
        },
        'optimizer': {
            'generator': optG.state_dict(),
            'discriminator': optD.state_dict()
        },
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_%03d.pth' % (checkpoint_path, epoch + 1))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
