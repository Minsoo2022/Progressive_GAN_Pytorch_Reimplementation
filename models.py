from networks import Generator, Discriminator

import torch
from torch import nn

class Progressive_GAN():
    def __init__(self, stage=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = Generator(stage).to(self.device)
        self.D = Discriminator(stage).to(self.device)
        self.optim_G = torch.optim.Adam(list(self.G.parameters()))
        self.optim_D = torch.optim.Adam(list(self.D.parameters()))
        self.GAN_Loss = nn.MSELoss()


    def stage_up(self):
        self.G.stage_up()
        self.D.stage_up()
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

    def train(self, real, alpha):
        real = real.to(self.device)
        batch_size = real.shape[0]
        latent_vector = torch.rand(batch_size, 512).to(self.device)
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
