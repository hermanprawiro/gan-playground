import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import vae
from utils.criterion import VAELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term for Adam (beta1)")
parser.add_argument("--beta2", type=float, default=0.999, help="Momentum term for Adam (beta2)")
parser.add_argument("--beta", type=float, default=1, help="KL Divergence weight beta (beta-VAE)")
parser.add_argument("--ndf", type=int, default=16, help="Base features multiplier for discriminator")
parser.add_argument("--ngf", type=int, default=16, help="Base features multiplier for generator")
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=128)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="vae")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")
# parser.add_argument("--data_root", type=str, default=R"E:\Datasets\MNIST")

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
        transforms.Resize(args.image_res),
        transforms.CenterCrop(args.image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args.image_ch, [0.5]*args.image_ch)
    ])

    dataset = torchvision.datasets.CelebA(args.data_root, split="all", transform=tfs, download=True)
    # dataset = torchvision.datasets.MNIST(args.data_root, train=True, transform=tfs, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netE = vae.Encoder(ndf=args.ndf, img_dim=args.image_ch, resolution=args.image_res, output_dim=args.latent_dim).to(device)
    netD = vae.Generator(ngf=args.ngf, img_dim=args.image_ch, resolution=args.image_res, z_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(list(netE.parameters()) + list(netD.parameters()), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    criterion = VAELoss(beta=args.beta).to(device)

    netE.train()
    netD.train()
    for epoch in range(args.n_epochs):
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)

            optimizer.zero_grad()

            mu, logvar = netE.encode(inputs)
            z = netE.reparameterize(mu, logvar)
            x_recon = netD(z)
            loss = criterion(mu, logvar, x_recon, inputs)

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f | Mu/Var: %.4f/%.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), loss.item(), mu.mean().item(), logvar.exp().mean().item()))

            if i % 50 == 0:
                outG = torch.cat([inputs[:32].cpu(), x_recon[:32].cpu()], dim=0)
                save_image(outG, '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), normalize=True)
        save_model((netE, netD), optimizer, epoch, args.checkpoint_path)            


def save_model(models, optimizers, epoch, checkpoint_path):
    netE, netD = models

    checkpoint = {
        'state_dict': {
            'encoder': netE.state_dict(),
            'decoder': netD.state_dict(),
        },
        'optimizer': optimizers.state_dict(),
        'epoch': epoch,
    }

    torch.save(checkpoint, '%s/checkpoint_%03d.pth' % (checkpoint_path, epoch + 1))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)