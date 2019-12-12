import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import acgan
from utils.criterion import GANLoss
from utils.misc import AverageMeter, accuracy, to_one_hot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--ndf", type=int, default=16, help="Base features multiplier for discriminator")
parser.add_argument("--ngf", type=int, default=16, help="Base features multiplier for generator")
parser.add_argument("--n_class", type=int, default=10)
parser.add_argument("--n_workers", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=64)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="acgan")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")
# parser.add_argument("--data_root", type=str, default=R"E:\Datasets\MNIST")

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
    # dataset = torchvision.datasets.MNIST(args.data_root, train=True, transform=tfs, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

    netG = acgan.Generator(z_dim=args.latent_dim, n_class=args.n_class, ngf=args.ngf, img_dim=args.image_ch).to(device)
    netD = acgan.Discriminator(n_class=args.n_class, ndf=args.ndf, img_dim=args.image_ch).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=4e-4, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    criterion_adv = GANLoss('vanilla', target_real_label=0.9, target_fake_label=0.1, target_fake_G_label=0.9).to(device)
    # criterion_aux = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()

    # fixed_noise = torch.randn(args.n_class * 8, args.latent_dim, device=device)
    # fixed_label = torch.arange(args.n_class, device=device).view(-1, 1).repeat(1, 8).flatten()
    # fixed_label = to_one_hot(fixed_label, args.n_class)
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    fixed_real = next(iter(dataloader))[0][:32].to(device)
    fixed_real, fixed_label = next(iter(dataloader))
    fixed_real = fixed_real[:64].to(device)
    fixed_label = fixed_label[:64].float().to(device)
    # fixed_label = torch.randint(0, 2, (64, args.n_class), dtype=torch.float, device=device) * 2 - 1

    save_image(fixed_real, '%s/real.jpg' % (args.result_path), normalize=True)

    netG.train()
    netD.train()
    for epoch in range(args.n_epochs):
        top1_real = AverageMeter()
        top1_fake = AverageMeter()

        for i, (img_real, lbl_real) in enumerate(dataloader):
            img_real = img_real.to(device)
            lbl_real = lbl_real.to(device)
            lbl_real = lbl_real.float()

            n_batch = img_real.shape[0]

            optD.zero_grad()
            # Discriminator (Real)
            outD_adv, outD_cls = netD(img_real)
            Dx = torch.sigmoid(outD_adv).mean().item()
            lossD_real_adv = criterion_adv(outD_adv, True)
            lossD_real_aux = criterion_aux(outD_cls, lbl_real)
            lossD_real = lossD_real_adv + lossD_real_aux * 0.5
            lossD_real.backward()

            # prec1 = accuracy(outD_cls, lbl_real)
            # top1_real.update(prec1[0], n_batch)

            # Generate Fake
            z = torch.randn(n_batch, args.latent_dim, device=device)
            # c = torch.randint(args.n_class, (n_batch,), device=device)
            # outG = netG(z, to_one_hot(c, args.n_class))
            # c = torch.randint(0, 2, (n_batch, args.n_class), device=device) * 2 - 1
            c = lbl_real.float()
            outG = netG(z, c)

            # Discriminator (Fake)
            outD_adv, outD_cls = netD(outG.detach())
            Dgz1 = torch.sigmoid(outD_adv).mean().item()
            lossD_fake_adv = criterion_adv(outD_adv, False)
            lossD_fake_aux = criterion_aux(outD_cls, c)
            lossD_fake = lossD_fake_adv + lossD_fake_aux * 0.5
            lossD_fake.backward()
            optD.step()

            lossD_adv = lossD_real_adv + lossD_fake_adv
            lossD_aux = lossD_real_aux + lossD_fake_aux
            # prec1 = accuracy(outD_cls, c)
            # top1_fake.update(prec1[0], n_batch)

            optG.zero_grad()
            # Generator
            outD_adv, outD_cls = netD(outG)
            Dgz2 = torch.sigmoid(outD_adv).mean().item()
            lossG_adv = criterion_adv(outD_adv, False, True)
            lossG_aux = criterion_aux(outD_cls, c)
            lossG = lossG_adv + lossG_aux * 0.5
            lossG.backward()
            optG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f / %.4f Loss_G: %.4f / %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch + 1, args.n_epochs, i, len(dataloader), lossD_adv.item(), lossD_aux.item(), lossG_adv.item(), lossG_aux.item(), Dx, Dgz1, Dgz2))

            if i % 50 == 0:
                outG = netG(fixed_noise, fixed_label).detach()
                save_image(outG, '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), normalize=True)
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
