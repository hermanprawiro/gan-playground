import torch
import torch.nn as nn
import torch.nn.functional as F

def G_arch(ngf=64, img_dim=3):
    arch = {}
    arch[32] = {
        'in_channels': [ngf * item for item in [8, 4]],
        'out_channels': [ngf * item for item in [4, 2]],
    }
    arch[64] = {
        'in_channels': [ngf * item for item in [16, 8, 4]],
        'out_channels': [ngf * item for item in [8, 4, 2]],
    }
    arch[128] = {
        'in_channels': [ngf * item for item in [16, 8, 4, 2]],
        'out_channels': [ngf * item for item in [8, 4, 2, 1]],
    }
    return arch

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, img_dim=3, resolution=64, bottom_width=4, init='N02', skip_init=False):
        super().__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.img_dim = img_dim
        self.resolution = resolution
        self.bottom_width = bottom_width
        self.init = init
        self.arch = G_arch(ngf=ngf, img_dim=img_dim)[resolution]

        self.linear = nn.Linear(z_dim, self.arch['in_channels'][0] * (bottom_width**2))
        
        self.blocks = nn.ModuleList()
        for idx in range(len(self.arch['in_channels'])):
            self.blocks.append(nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(self.arch['in_channels'][idx], self.arch['out_channels'][idx], 4, stride=2, padding=1)),
                nn.BatchNorm2d(self.arch['out_channels'][idx]),
                nn.ReLU(True)
            ))
        self.out_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(self.arch['out_channels'][-1], img_dim, 4, stride=2, padding=1)),
            nn.Tanh()
        )

        if not skip_init:
            self.init_weights()

    def forward(self, z):
        h = F.relu(self.linear(z), True)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        for idx, block in enumerate(self.blocks):
            h = block(h)
        out = self.out_layer(h)

        return out

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.ConvTranspose2d) 
            or isinstance(module, nn.Linear)):
                if self.init == 'ortho':
                    nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
            # if (isinstance(module, nn.BatchNorm2d)):
            #     nn.init.normal_(module.weight, 1.0, 0.02)
            
            self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

def D_arch(ndf=64, img_dim=3):
    arch = {}
    arch[32] = {
        'in_channels': [img_dim] + [ndf * item for item in [2, 4]],
        'out_channels': [ndf * item for item in [2, 4, 8]],
    }
    arch[64] = {
        'in_channels': [img_dim] + [ndf * item for item in [2, 4, 8]],
        'out_channels': [ndf * item for item in [2, 4, 8, 16]],
    }
    arch[128] = {
        'in_channels': [img_dim] + [ndf * item for item in [1, 2, 4, 8]],
        'out_channels': [ndf * item for item in [1, 2, 4, 8, 16]],
    }
    return arch

class Discriminator(nn.Module):
    def __init__(self, z_dim=100, ndf=64, img_dim=3, resolution=64, output_dim=1, init='N02', skip_init=False):
        super().__init__()
        self.ndf = ndf
        self.img_dim = img_dim
        self.resolution = resolution
        self.init = init
        self.arch = D_arch(ndf=ndf, img_dim=img_dim)[resolution]

        # Submodule F(x)
        self.blocks = nn.ModuleList()
        for idx in range(len(self.arch['in_channels'])):
            block = []
            block.append(nn.utils.spectral_norm(nn.Conv2d(self.arch['in_channels'][idx], self.arch['out_channels'][idx], 4, stride=2, padding=1)))
            if idx != 0:
                block.append(nn.BatchNorm2d(self.arch['out_channels'][idx]))
            block.append(nn.LeakyReLU(0.2, True))
            self.blocks.append(nn.Sequential(*block))
        last_hidden = self.arch['out_channels'][-1]
        self.feat_f = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(last_hidden, ndf * 8)),
            nn.LeakyReLU(0.2, True)
        )
        self.out_f = nn.Linear(ndf * 8, 1)

        # Submodule H(z)
        self.feat_h = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, ndf * 8)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 8)),
            nn.LeakyReLU(0.2, True),
        )
        self.out_h = nn.Linear(ndf * 8, 1)

        # Submodule J(F(x), H(z))
        self.out_j = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(ndf * 16, ndf * 8)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 8)),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf * 8, 1)
        )

        if not skip_init:
            self.init_weights()

    def forward(self, x, z):
        f = x
        for idx, block in enumerate(self.blocks):
            f = block(f)
        f = torch.sum(f, dim=[2, 3]) # Global sum pooling
        f = self.feat_f(f)
        s_f = self.out_f(f)
        
        h = self.feat_h(z)
        s_h = self.out_h(h)

        s_j = self.out_j(torch.cat([f, h], 1))

        return s_f, s_h, s_j

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
            or isinstance(module, nn.Linear)):
                if self.init == 'ortho':
                    nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
            # if (isinstance(module, nn.BatchNorm2d)):
            #     nn.init.normal_(module.weight, 1.0, 0.02)
            
            self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)


def E_arch(ndf=64, img_dim=3):
    arch = {}
    arch[32] = {
        'in_channels': [img_dim] + [ndf * item for item in [2, 4]],
        'out_channels': [ndf * item for item in [2, 4, 8]],
    }
    arch[64] = {
        'in_channels': [img_dim] + [ndf * item for item in [2, 4, 8]],
        'out_channels': [ndf * item for item in [2, 4, 8, 16]],
    }
    arch[128] = {
        'in_channels': [img_dim] + [ndf * item for item in [1, 2, 4, 8]],
        'out_channels': [ndf * item for item in [1, 2, 4, 8, 16]],
    }
    return arch

class Encoder(nn.Module):
    def __init__(self, ndf=64, img_dim=3, resolution=64, output_dim=100, init='N02', skip_init=False):
        super().__init__()
        self.ndf = ndf
        self.img_dim = img_dim
        self.resolution = resolution
        self.init = init
        self.arch = D_arch(ndf=ndf, img_dim=img_dim)[resolution]

        self.blocks = nn.ModuleList()
        for idx in range(len(self.arch['in_channels'])):
            self.blocks.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(self.arch['in_channels'][idx], self.arch['out_channels'][idx], 4, stride=2, padding=1)),
                nn.BatchNorm2d(self.arch['out_channels'][idx]),
                nn.ReLU(True)
            ))
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.arch['out_channels'][-1], ndf * 8)),
            nn.ReLU(True)
        )
        self.out_mu = nn.Linear(ndf * 8, output_dim)
        self.out_logvar = nn.Linear(ndf * 8, output_dim)

        if not skip_init:
            self.init_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        out = self.reparameterize(mu, logvar)

        return out

    def reparameterize(self, mu, logvar):
        std = F.softplus(logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        h = x
        for idx, block in enumerate(self.blocks):
            h = block(h)
        h = torch.sum(h, dim=[2, 3]) # Global sum pooling
        h = self.fc(h)
        
        mu = self.out_mu(h)
        logvar = self.out_logvar(h)
        return mu, logvar

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
            or isinstance(module, nn.Linear)):
                if self.init == 'ortho':
                    nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
            # if (isinstance(module, nn.BatchNorm2d)):
            #     nn.init.normal_(module.weight, 1.0, 0.02)
            
            self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for E''s initialized parameters: %d' % self.param_count)

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()
    netE = Encoder()

    z = torch.randn(4, 100)
    fake = netG(z)
    logits_f, logits_h, logits_j = netD(fake, z)
    print(fake.shape)
    print(logits_f.shape, logits_h.shape, logits_j.shape)
    enc_z = netE(fake)
    logits_f, logits_h, logits_j = netD(fake, enc_z)
    print(enc_z.shape)
    print(logits_f.shape, logits_h.shape, logits_j.shape)