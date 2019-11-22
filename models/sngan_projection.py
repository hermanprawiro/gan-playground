import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

# From https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/sn_projection_cgan_64x64_143c.ipynb
class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        nn.init.zeros_(self.c1.bias)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        nn.init.zeros_(self.c2.bias)
        if n_classes > 0:
            self.b1 = ConditionalBatchNorm2d(in_channels, n_classes)
            self.b2 = ConditionalBatchNorm2d(hidden_channels, n_classes)
        else:
            self.b1 = nn.BatchNorm2d(in_channels)
            self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight)
            nn.init.zeros_(self.c_sc.bias)

    def forward(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
        h = self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            sc = self.c_sc(x)
        else:
            sc = x
        return h + sc

class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, img_dim=3, activation=F.relu, n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        self.block2 = ResGenBlock(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = ResGenBlock(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = ResGenBlock(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = ResGenBlock(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b6 = nn.BatchNorm2d(ch)
        self.l6 = nn.Conv2d(ch, img_dim, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l6.weight)
        nn.init.zeros_(self.l6.bias)

    def forward(self, z, y):
        h = z
        h = self.l1(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.l6(h))

        return h

class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        nn.init.zeros_(self.c1.bias)
        nn.utils.spectral_norm(self.c1)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        nn.init.zeros_(self.c2.bias)
        nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight)
            nn.init.zeros_(self.c_sc.bias)
            nn.utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.learnable_sc:
            sc = self.c_sc(x)
            if self.downsample:
                sc = F.avg_pool2d(sc, 2)
        else:
            sc = x
        return h + sc

class ResDisOptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        nn.init.zeros_(self.c1.bias)
        nn.utils.spectral_norm(self.c1)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        nn.init.zeros_(self.c2.bias)
        nn.utils.spectral_norm(self.c2)
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        nn.init.xavier_uniform_(self.c_sc.weight)
        nn.init.zeros_(self.c_sc.bias)
        nn.utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        sc = self.c_sc(x)
        sc = F.avg_pool2d(sc, 2)
        return h + sc

class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, img_dim=3, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.block1 = ResDisOptimizedBlock(img_dim, ch)
        self.block2 = ResDisBlock(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = ResDisBlock(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = ResDisBlock(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = ResDisBlock(ch * 8, ch * 16, activation=activation, downsample=True)
        self.l6 = nn.Linear(ch * 16, 1)
        nn.init.xavier_uniform_(self.l6.weight)
        nn.init.zeros_(self.l6.bias)
        nn.utils.spectral_norm(self.l6)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, ch * 16)
            nn.init.xavier_uniform_(self.l_y.weight)
            nn.utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output

if __name__ == "__main__":
    netG = ResNetGenerator(n_classes=10)
    netD = SNResNetProjectionDiscriminator(n_classes=10)

    z = torch.randn(4, 128)
    c = torch.randint(10, (4,))
    fake = netG(z, c)
    logits = netD(fake, c)
    print(fake.shape)
    print(logits.shape)