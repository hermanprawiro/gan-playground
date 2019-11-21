import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_class=10, ngf=64, img_dim=3):
        super().__init__()

        self.class_embed = nn.Embedding(n_class, latent_dim)

        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(latent_dim, ngf * 16, 4)), # (n, ngf * 16, 4, 4)
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=1)), # (n, ngf * 8, 8, 8)
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1)), # (n, ngf * 4, 16, 16)
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1)), # (n, ngf * 2, 32, 32)
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1)), # (n, ngf, 64, 64)
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_out = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf, img_dim, 4, stride=2, padding=1)), # (n, img_dim, 128, 128)
            nn.Tanh(),
        )

    def forward(self, z, c):
        z = z * self.class_embed(c)
        z = z[..., None, None]
        
        out = self.conv1(z)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv_out(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, n_class=10, ndf=64, img_dim=3):
        super().__init__()

        # (n, img_dim, 128, 128)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_dim, ndf, 4, stride=2, padding=1)), # (n, ndf, 64, 64)
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1)), # (n, ndf * 2, 32, 32)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)), # (n, ndf * 4, 16, 16)
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1)), # (n, ndf * 8, 8, 8)
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1)), # (n, ndf * 16, 4, 4)
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, True),
        )

        self.out_adv = nn.utils.spectral_norm(nn.Conv2d(ndf * 16, 1, 4))
        self.out_cls = nn.utils.spectral_norm(nn.Conv2d(ndf * 16, n_class, 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out_adv = self.out_adv(out).flatten(start_dim=1)
        out_cls = self.out_cls(out).flatten(start_dim=1)

        return out_adv, out_cls

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()

    z = torch.randn(4, 100)
    c = torch.randint(10, (4,))
    fake = netG(z, c)
    logits_adv, logits_cls = netD(fake)
    print(fake.shape)
    print(logits_adv.shape, logits_cls.shape)