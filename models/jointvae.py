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

        features_dim_in = self.arch['in_channels'][0]
        self.latent_to_features = nn.Sequential(
            nn.Linear(z_dim, features_dim_in),
            nn.ReLU(True),
            nn.Linear(features_dim_in, features_dim_in * (bottom_width**2)),
            nn.ReLU(True),
        )
        
        self.blocks = nn.ModuleList()
        for idx in range(len(self.arch['in_channels'])):
            self.blocks.append(nn.Sequential(
                nn.ConvTranspose2d(self.arch['in_channels'][idx], self.arch['out_channels'][idx], 4, stride=2, padding=1),
                nn.BatchNorm2d(self.arch['out_channels'][idx]),
                nn.ReLU(True)
            ))
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(self.arch['out_channels'][-1], img_dim, 4, stride=2, padding=1),
            nn.Tanh()
        )

        if not skip_init:
            self.init_weights()

    def forward(self, z):
        h = self.latent_to_features(z)
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

class Encoder(nn.Module):
    def __init__(self, ndf=64, img_dim=3, resolution=64, bottom_width=4, latent_cont_dim=100, latent_disc_dims=[10], temperature=.67, init='N02', skip_init=False):
        super().__init__()
        self.ndf = ndf
        self.img_dim = img_dim
        self.resolution = resolution
        self.init = init
        self.arch = D_arch(ndf=ndf, img_dim=img_dim)[resolution]

        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dims = latent_disc_dims
        self.temperature = temperature # For gumbel softmax distribution

        self.blocks = nn.ModuleList()
        for idx in range(len(self.arch['in_channels'])):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(self.arch['in_channels'][idx], self.arch['out_channels'][idx], 4, stride=2, padding=1),
                nn.BatchNorm2d(self.arch['out_channels'][idx]),
                nn.ReLU(True)
            ))

        features_dim_out = self.arch['out_channels'][-1]
        self.features_to_hidden = nn.Sequential(
            nn.Linear(features_dim_out * (bottom_width ** 2), features_dim_out),
            nn.ReLU(True),
            nn.Linear(features_dim_out, features_dim_out),
            nn.ReLU(True)
        )

        self.out_mu = nn.Linear(features_dim_out, latent_cont_dim)
        self.out_logvar = nn.Linear(features_dim_out, latent_cont_dim)
        self.out_alphas = nn.ModuleList()
        for latent_disc_dim in self.latent_disc_dims:
            self.out_alphas.append(nn.Linear(features_dim_out, latent_disc_dim))

        if not skip_init:
            self.init_weights()

    def forward(self, x):
        mu, logvar, alphas = self.encode(x)
        out = self.reparameterize(mu, logvar, alphas)

        return out

    def reparameterize(self, mu, logvar, alphas):
        z = []
        z.append(self.sample_gaussian(mu, logvar))
        for alpha in alphas:
            z.append(self.sample_gumbel_softmax(alpha))

        return torch.cat(z, dim=1)

    def encode(self, x):
        h = x
        for idx, block in enumerate(self.blocks):
            h = block(h)
        # h = torch.sum(h, dim=[2, 3]) # Global sum pooling
        h = self.features_to_hidden(h.flatten(1))

        mu = self.out_mu(h)
        logvar = self.out_logvar(h)
        alphas = []
        for out_alpha in self.out_alphas:
            alphas.append(torch.softmax(out_alpha(h), dim=-1))

        return mu, logvar, alphas

    def sample_gaussian(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            # std = F.softplus(logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def sample_gumbel_softmax(self, alpha):
        if self.training:
            # return F.gumbel_softmax(alpha, tau=self.temperature, hard=False)
            unif = torch.rand(alpha.size()).cuda()
            # if self.use_cuda:
            #     unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + 1e-12) + 1e-12)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + 1e-12)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # return F.gumbel_softmax(alpha, tau=self.temperature, hard=True)
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            # if self.use_cuda:
            #     one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

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
    netG = Generator(resolution=64, ngf=32, z_dim=20)
    netE = Encoder(resolution=64, ndf=32, latent_cont_dim=10, latent_disc_dims=[10])

    print(netG)
    print(netE)

    x = torch.randn(2, 3, 64, 64)
    z = netE(x)
    print(z)
    print(z.shape)
    x_recon = netG(z)
    print(x_recon.shape)