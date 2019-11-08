import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_dim=3):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512*4*4) # (n, 512, 4, 4)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), # (n, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), # (n, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False), # (n, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, out_dim, 4, stride=2, padding=1, bias=False), # (n, out_dim, 64, 64)
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z).view(-1, 512, 4, 4)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()

        # (n, in_dim, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, 4, stride=2, padding=1), # (n, 64, 32, 32)
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False), # (n, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False), # (n, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False), # (n, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4), # (n, 1, 1, 1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.flatten(start_dim=1)

        return out

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()

    z = torch.randn(4, 100)
    fake = netG(z)
    logits = netD(fake)
    print(fake.shape)
    print(logits.shape)