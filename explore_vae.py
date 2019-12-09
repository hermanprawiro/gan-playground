import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

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
parser.add_argument("--n_workers", type=int, default=0)
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--image_ch", type=int, default=3)
parser.add_argument("--image_res", type=int, default=128)
parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
parser.add_argument("--result_path", type=str, default="results")
parser.add_argument("--save_name", type=str, default="vae")
parser.add_argument("--data_root", type=str, default=R"E:\Datasets\CelebA")
# parser.add_argument("--data_root", type=str, default=R"E:\Datasets\MNIST")

args = parser.parse_args()

args.save_name = 'vae_celeba_b512_z10'

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
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)

netE = vae.Encoder(ndf=args.ndf, img_dim=args.image_ch, resolution=args.image_res, output_dim=args.latent_dim).to(device)
netD = vae.Generator(ngf=args.ngf, img_dim=args.image_ch, resolution=args.image_res, z_dim=args.latent_dim).to(device)

optimizer = torch.optim.Adam(list(netE.parameters()) + list(netD.parameters()), lr=args.learning_rate, betas=(args.beta1, args.beta2))

checkpoint = torch.load('%s/checkpoint_%03d.pth' % (args.checkpoint_path, 20), map_location=torch.device('cpu'))
netE.load_state_dict(checkpoint['state_dict']['encoder'])
netD.load_state_dict(checkpoint['state_dict']['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])


netE.eval()
netD.eval()

inputs, labels = next(iter(dataloader))
inputs = inputs.to(device)

n_interp = 3
interp_grad = torch.linspace(0, 1, n_interp)
idx = torch.arange(args.latent_dim)
idx = torch.zeros((args.latent_dim, args.latent_dim)).scatter_(1, idx.view(-1, 1), 1).unsqueeze(-1).expand(-1, -1, n_interp)
interp_weight = (idx * interp_grad).to(device)

idx_start = 1
idx_end = 60

with torch.no_grad():
    mu, logvar = netE.encode(inputs)
    z = netE.reparameterize(mu, logvar)
    x_recon = netD(z)

print(mu[idx_start], logvar[idx_start].exp())
print(mu[idx_end], logvar[idx_end].exp())
z_start = z[idx_start]
z_end = z[idx_end]

z_interp = torch.empty((args.latent_dim + 1, n_interp, args.latent_dim), dtype=torch.float, device=device)
for i in range(args.latent_dim):
    for j in range(n_interp):
        z_interp[i, j] = torch.lerp(z_start, z_end, interp_weight[i, :, j])
    
for j, val in enumerate(interp_grad):
    z_interp[-1, j] = torch.lerp(z_start, z_end, val.item())

with torch.no_grad():
    x_interp = netD(z_interp.view(-1, args.latent_dim))

print(z_interp.shape)

inputs_row = torch.zeros((n_interp,) + x_interp.shape[1:])
inputs_row[0] = inputs[idx_start]
inputs_row[-1] = inputs[idx_end]
x_interp = torch.cat([x_interp.cpu(), inputs_row], dim=0)

# plt.imshow(make_grid(inputs.cpu(), normalize=True).permute(1, 2, 0))
# plt.show()

# plt.imshow(make_grid(x_recon.cpu(), normalize=True).permute(1, 2, 0))
# plt.show()

plt.imshow(make_grid(x_interp.cpu(), normalize=True, nrow=n_interp).permute(1, 2, 0))
plt.show()

# for epoch in range(args.n_epochs):
#     for i, (inputs, _) in enumerate(dataloader):
#         inputs = inputs.to(device)

#         optimizer.zero_grad()

#         mu, logvar = netE.encode(inputs)
#         z = netE.reparameterize(mu, logvar)
#         x_recon = netD(z)
#         loss = criterion(mu, logvar, x_recon, inputs)

#         loss.backward()
#         optimizer.step()

#         if i % 50 == 0:
#             print('[%d/%d][%d/%d] Loss: %.4f | Mu/Var: %.4f/%.4f'
#                 % (epoch + 1, args.n_epochs, i, len(dataloader), loss.item(), mu.mean().item(), logvar.exp().mean().item()))

#         if i % 50 == 0:
#             outG = torch.cat([inputs[:32].cpu(), x_recon[:32].cpu()], dim=0)
#             save_image(outG, '%s/fake_epoch%03d_%04d.jpg' % (args.result_path, epoch + 1, i + 1), normalize=True)
#     save_model((netE, netD), optimizer, epoch, args.checkpoint_path)