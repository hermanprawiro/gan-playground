import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, target_fake_G_label=1.0):
        super().__init__()

        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('fake_G_label', torch.tensor(target_fake_G_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError('GAN mode %s is not implemented' % gan_mode)
    
    def _get_target_tensor(self, prediction, is_real, is_generator=False):
        if is_real:
            target_tensor = self.real_label
        elif is_generator:
            target_tensor = self.fake_G_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, is_real, is_generator=False):
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self._get_target_tensor(prediction, is_real, is_generator)
            loss = self.loss(prediction, target_tensor)
        else:
            if is_real:
                loss = nn.functional.relu(1. - prediction).mean()
            elif is_generator:
                loss = - prediction.mean()
            else:
                loss = nn.functional.relu(1. + prediction).mean()
        return loss

class VAELoss(nn.Module):
    def __init__(self, recon_mode='l2', beta=1):
        super().__init__()

        recon_dict = {
            'bce': nn.BCELoss(reduction='none'),
            'l1': nn.L1Loss(reduction='none'),
            'l2': nn.MSELoss(reduction='none'),
            'smoothl1': nn.SmoothL1Loss(reduction='none'),
            'none': None,
        }
        if recon_mode not in recon_dict.keys():
            raise NotImplementedError('Reconstruction mode %s is not implemented' % recon_mode)
        # Normalized beta = beta * latent_dim / input_dim
        self.beta = beta
        self.recon_criterion = recon_dict[recon_mode]

    def forward(self, mu, logvar, recon_x=None, target_x=None):
        if self.recon_criterion is not None:
            recon_loss = self.recon_criterion(recon_x, target_x).sum((1, 2, 3)).mean()
        else:
            recon_loss = 0
        
        # KL Divergence => 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()

        return recon_loss + self.beta * kld_loss