from networks import Generator, Discriminator

import torch
from torch import nn
import os

class Progressive_GAN():
    def __init__(self, opt, config):
        stage=opt.init_stage
        self.ch_Latent = config['ch_Latent']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = Generator(stage, self.ch_Latent, config['depth_list']).to(self.device)
        self.D = Discriminator(stage, self.ch_Latent, config['depth_list']).to(self.device)
        self.optim_G = torch.optim.Adam(list(self.G.parameters()), opt.G_lr, betas=[0.0,0.99])
        self.optim_D = torch.optim.Adam(list(self.D.parameters()), opt.D_lr, betas=[0.0,0.99])
        self.GAN_Loss = nn.MSELoss()


    def stage_up(self):
        self.G.stage_up()
        self.D.stage_up()
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        print('**************************Stage-Up***********************************')

    def train(self, real, alpha):
        real = real.to(self.device)
        batch_size = real.shape[0]
        print(batch_size)
        latent_vector = torch.randn(batch_size, self.ch_Latent).to(self.device)
        real_label = torch.ones(batch_size).to(self.device)
        fake_label = torch.zeros(batch_size).to(self.device)

        fake = self.G(latent_vector, alpha)
        real_pred = self.D(real, alpha)
        fake_pred = self.D(fake.detach(), alpha)
        loss_D_real = self.GAN_Loss(real_pred, real_label)
        loss_D_fake = self.GAN_Loss(fake_pred, fake_label)
        loss_D = loss_D_real + loss_D_fake

        self.D.zero_grad()
        loss_D.backward()
        self.optim_D.step()

        #########################################################################################
        G_pred = self.D(fake,alpha)
        loss_G = self.GAN_Loss(G_pred, real_label)
        self.G.zero_grad()
        loss_G.backward()
        self.optim_G.step()
        losses={}
        losses['real_pred'] = real_pred
        losses['fake_pred'] = fake_pred
        losses['loss_D'] = loss_D
        losses['loss_G'] = loss_G

        return losses, fake

    def save(self, opt, iteration, device):
        path = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.isdir(path) :
            os.makedirs(path,exist_ok=True)
        torch.save({
            'iteration' : iteration,
            'G' : self.G.cpu().state_dict(),
            'D' : self.D.cpu().state_dict(),
            'optim_G' : self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
        }, os.path.join(path,f'model_{iteration}'))
        self.G.to(device)
        self.D.to(device)