import argparse
import os
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import infogan
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
parser.add_argument("--n_disc_update", type=int, default=1)
parser.add_argument("--n_gen_update", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--noise_dim", type=int, default=62)
parser.add_argument("--cont_dim", type=int, default=2)
parser.add_argument("--cat_dim", type=int, default=10)
parser.add_argument("--lambda_mi_cont", type=float, default=0.01)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=64)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="infogan_mnist")
# parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\MNIST")

def to_one_hot(y, n_class):
    return torch.zeros((y.size(0), n_class), device=y.device).scatter_(1, y.view(-1, 1), 1)

def gaussian_log_loss(value, mu, var):
    TINY = 1e-8
    eps = (value - mu) / (var + TINY)
    logli = -0.5 * torch.log(2 * torch.tensor(math.pi)) - torch.log(var + TINY) - 0.5 * eps.pow(2)
    return logli.sum(1).mean().mul(-1)

def main(args):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    args.latent_dim = args.noise_dim + args.cont_dim + args.cat_dim

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

    # dataset = torchvision.datasets.CelebA(args.data_root, split="all", transform=tfs, download=True)
    dataset = torchvision.datasets.MNIST(args.data_root, train=True, transform=tfs, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netG = infogan.Generator(z_dim=args.latent_dim, ngf=args.ngf, img_dim=args.image_ch, resolution=args.image_res).to(device)
    netD = infogan.Discriminator(ndf=args.ndf, img_dim=args.image_ch, resolution=args.image_res).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=4e-4, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))
    optMI = torch.optim.Adam(list(netG.parameters()) + list(netD.parameters()), lr=1e-4, betas=(args.beta1, args.beta2))

    criterion = GANLoss('vanilla', target_real_label=0.9, target_fake_label=0.0, target_fake_G_label=0.9).to(device)
    criterion_cat = nn.CrossEntropyLoss()

    # Generate Fixed Latent for Visualization
    fixed_noise = torch.randn(args.cat_dim * 10 * 2, args.noise_dim, device=device)
    fixed_cat = to_one_hot(torch.arange(args.cat_dim, device=device).view(-1, 1).repeat(2, 10).flatten(), args.cat_dim)
    fixed_cont_var = torch.linspace(-3., 3., 10, device=device).view(-1, 1).repeat(10, 1)
    fixed_cont_static = torch.zeros_like(fixed_cont_var)
    fixed_latent = torch.cat([fixed_noise, fixed_cat, torch.cat([fixed_cont_var, fixed_cont_static], dim=0), torch.cat([fixed_cont_static, fixed_cont_var], dim=0)], dim=1)

    # Prior continuous distribution (Gaussian)
    prior_mu = torch.zeros(args.cont_dim, device=device)
    prior_var = torch.ones(args.cont_dim, device=device)
    prior_dist = torch.distributions.Normal(prior_mu, prior_var)

    netG.train()
    netD.train()
    for epoch in range(args.n_epochs):
        for i, (x_real, labels) in enumerate(dataloader):
            x_real = x_real.to(device)
            labels = labels.to(device)

            # Generate random latent
            n_batch = x_real.shape[0]
            z_noise = torch.randn(n_batch, args.noise_dim, device=device)
            z_labels = torch.randint(args.cat_dim, (n_batch,), device=device)
            z_cat = to_one_hot(z_labels, args.cat_dim)
            z_cont = prior_dist.sample((n_batch,)).to(device)
            z = torch.cat([z_noise, z_cat, z_cont], dim=1)

            # Generate Fake
            outG = netG(z)

            optG.zero_grad()
            # Generator
            outD, _, _, _ = netD(outG)
            Dgz2 = outD.mean().item()
            lossG = criterion(outD, False, True)
            lossG.backward()
            optG.step()

            optD.zero_grad()
            # Discriminator (Real)
            outD, _, _, _ = netD(x_real)
            Dx = outD.mean().item()
            lossD_real = criterion(outD, True)

            # Discriminator (Fake)
            outD, _, _, _ = netD(outG.detach())
            Dgz1 = outD.mean().item()
            lossD_fake = criterion(outD, False)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # Mutual Information
            optMI.zero_grad()
            # Enforce real sample's categorical distribution
            # _, outCat, _, _ = netD(x_real)
            # lossMI_cat_real = criterion_cat(outCat, labels)
            # Original InfoGAN Mutual Information Losses
            outG = netG(z)
            _, outCat, outMu, outLogvar = netD(outG)
            lossMI_cat = criterion_cat(outCat, z_labels)
            lossMI_cont = gaussian_log_loss(z_cont, outMu, outLogvar.exp()) - gaussian_log_loss(z_cont, prior_mu, prior_var)
            # lossMI = lossMI_cat_real + lossMI_cat + lossMI_cont*args.lambda_mi_cont
            lossMI = lossMI_cat + lossMI_cont*args.lambda_mi_cont
            lossMI.backward()
            optMI.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_MI: %.4f D(x): %.4f D(G(z)): %.4f / %.4f | Mu / Var: %.4f %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), lossD.item(), lossG.item(), lossMI.item(), Dx, Dgz1, Dgz2, outMu.mean().item(), outLogvar.exp().mean().item()))

            if i % 50 == 0:
                outG = netG(fixed_latent).detach()
                save_image(outG, '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), nrow=10)
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
