import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_dim=3):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(latent_dim, 512, 4, bias=False)), # (n, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)), # (n, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)), # (n, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)), # (n, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, out_dim, 4, stride=2, padding=1), # (n, out_dim, 64, 64)
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.conv0(z)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, in_dim=3):
        super().__init__()

        # (n, in_dim, 64, 64)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_dim, 64, 4, stride=2, padding=1)), # (n, 64, 32, 32)
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)), # (n, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)), # (n, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)), # (n, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 4, bias=False)), # (n, 512, 1, 1)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.conv1_z = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2_z = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2, True),
        )

        self.conv1_joint = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2_joint = nn.Conv2d(512, 1, 1)

    def forward(self, x, z):
        # D(x)
        h_x = self.conv1(x)
        h_x = self.conv2(h_x)
        h_x = self.conv3(h_x)
        h_x = self.conv4(h_x)
        h_x = self.conv5(h_x)
        # D(z)
        h_z = self.conv1_z(z)
        h_z = self.conv2_z(h_z)
        # Joint
        out = self.conv1_joint(torch.cat([h_x, h_z], dim=1))
        out = self.conv2_joint(out)
        
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim=100, in_dim=3):
        super().__init__()

        # (n, in_dim, 64, 64)
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_dim, 64, 4, stride=2, padding=1, bias=False)), # (n, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)), # (n, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)), # (n, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)), # (n, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 4, bias=False)), # (n, 512, 1, 1)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv6 = nn.Conv2d(512, latent_dim, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        return out

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()
    netE = Encoder()

    z = torch.randn(4, 100, 1, 1)
    fake = netG(z)
    logits = netD(fake, z)
    print(fake.shape)
    print(logits.shape)
    enc_z = netE(fake)
    logits_e = netD(fake, enc_z)
    print(logits_e.shape)