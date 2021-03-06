import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, ndf=64, img_dim=3):
        super().__init__()
        self.ndf = ndf
        self.img_dim = img_dim

        self.gen_z = nn.Linear(ndf * 2, ndf * 16 * 4 * 4)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 16, ndf * 8, 4, stride=2, padding=1, bias=False), # (n, ndf * 8, 8, 8)
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, stride=2, padding=1, bias=False), # (n, ndf * 4, 16, 16)
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, stride=2, padding=1, bias=False), # (n, ndf * 2, 32, 32)
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, 4, stride=2, padding=1, bias=False), # (n, ndf, 64, 64)
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(ndf, img_dim, 4, stride=2, padding=1), # (n, out_dim, 128, 128)
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.gen_z(z).view(-1, self.ndf * 16, 4, 4)

        out = self.conv1(z)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out

class Encoder(nn.Module):
    def __init__(self, ndf=64, img_dim=3):
        super().__init__()
        self.ndf = ndf
        self.img_dim = img_dim

        # (n, img_dim, 128, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_dim, ndf, 4, stride=2, padding=1, bias=False), # (n, ndf, 64, 64)
            nn.BatchNorm2d(ndf),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False), # (n, ndf * 2, 32, 32)
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False), # (n, ndf * 4, 16, 16)
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False), # (n, ndf * 8, 8, 8)
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False), # (n, ndf * 16, 4, 4)
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 16, ndf * 8),
            nn.ReLU(True),
            nn.Linear(ndf * 8, ndf * 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = torch.sum(out, dim=[2, 3]) # Global sum pooling

        out = self.fc(out)

        return out

if __name__ == "__main__":
    netE = Encoder()
    netG = Decoder()

    x_real = torch.randn(1, 3, 128, 128)
    z = netE(x_real)
    x_recon = netG(z)

    print(z.shape)
    print(x_recon.shape)