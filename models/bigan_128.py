import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, img_dim=3):
        super().__init__()

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

    def forward(self, z):
        z = z[..., None, None]
        out = self.conv1(z)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv_out(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, ndf=64, img_dim=3):
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

        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, ndf * 16),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(ndf * 16, ndf * 16),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
        )

        self.fc_h = nn.Sequential(
            nn.Linear(ndf * 16, ndf * 16),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(ndf * 16, ndf * 16),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
        )

        self.fc_joint = nn.Sequential(
            nn.Linear(ndf * 32, ndf * 32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(ndf * 32, ndf * 32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(ndf * 32, 1)
        )

    def forward(self, x, z):
        # D(x)
        h_x = self.conv1(x)
        h_x = self.conv2(h_x)
        h_x = self.conv3(h_x)
        h_x = self.conv4(h_x)
        h_x = self.conv5(h_x)
        h_x = torch.sum(h_x, [2, 3]) # Global sum pooling

        # D(z)
        h_z = self.fc_z(z)
        # Joint
        out = self.fc_joint(torch.cat([h_x, h_z], dim=1))
        
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim=100, ndf=64, img_dim=3):
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
        self.fc = nn.Linear(ndf * 16, latent_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = torch.sum(out, [2, 3])
        out = self.fc(out)

        return out

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()
    netE = Encoder()

    z = torch.randn(4, 100)
    fake = netG(z)
    logits = netD(fake, z)
    print(fake.shape)
    print(logits.shape)
    enc_z = netE(fake)
    logits_e = netD(fake, enc_z)
    print(logits_e.shape)