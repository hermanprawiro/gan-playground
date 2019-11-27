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
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/dcgan")
parser.add_argument("--result_path", type=str, default="results/dcgan")
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

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    tfs = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = torchvision.datasets.CelebA(args.data_root, split="all", transform=tfs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netG = dcgan_specnorm.Generator(latent_dim=args.latent_dim, out_dim=args.image_ch).to(device)
    netD = dcgan_specnorm.Discriminator(in_dim=args.image_ch).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optG = torch.optim.Adam(netG.parameters(), lr=4e-4, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    criterion = GANLoss('vanilla', target_real_label=0.0, target_fake_label=1.0, target_fake_G_label=0.0).to(device)

    fixed_noise = torch.randn(args.batch_size, args.latent_dim, device=device)

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

            if i % 200 == 0:
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
