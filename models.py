from networks import Generator, Discriminator
import torch
from torch import nn
import os

class Progressive_GAN():
    def __init__(self, opt, config):
        stage=opt.init_stage
        self.ch_Latent = config['ch_Latent']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GAN_type = opt.GAN_type
        self.G = Generator(stage, self.ch_Latent, config['depth_list'], opt.Norm, opt.Equalized).to(self.device)
        self.D = Discriminator(stage, self.ch_Latent, config['depth_list'], opt.Norm, opt.Equalized).to(self.device)
        self.optim_G = torch.optim.Adam(list(self.G.parameters()), opt.G_lr, betas=[0.0,0.99])
        self.optim_D = torch.optim.Adam(list(self.D.parameters()), opt.D_lr, betas=[0.0,0.99])
        self.init_stage()
        if self.GAN_type == 'LSGAN' :
            self.GAN_Loss = nn.MSELoss()
        elif self.GAN_type == 'WGAN_GP':
            self.GAN_Loss = GPLoss(self.device)
        if len(opt.gpu_ids.split(',')) > 1 :
            self.multi_gpu()

    def init_stage(self):
        self.G.stage_init()
        self.D.stage_init()
        self.G.to(self.device)
        self.D.to(self.device)

    def multi_gpu(self):
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)
        #self.GAN_Loss = nn.DataParallel(self.GAN_Loss)

    def single_gpu(self):
        self.G = self.G.module
        self.D = self.D.module

    def stage_up(self):
        if hasattr(self.G,'module') :
            multi = True
            self.single_gpu()
        else :
            multi = False
        self.G.stage_up()
        self.D.stage_up()
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        if multi :
            self.multi_gpu()
        print('**************************Stage-Up***********************************')

    def train(self, real, alpha):
        real = real.to(self.device)
        batch_size = real.shape[0]
        latent_vector = torch.randn(batch_size, self.ch_Latent).to(self.device)

        fake = self.G(latent_vector, alpha)
        real_pred = self.D(real, alpha)
        fake_pred = self.D(fake.detach(), alpha)
        if self.GAN_type == 'LSGAN':
            real_label = torch.ones(batch_size).to(self.device)
            fake_label = torch.zeros(batch_size).to(self.device)
            loss_D_real = self.GAN_Loss(real_pred, real_label)
            loss_D_fake = self.GAN_Loss(fake_pred, fake_label)
            loss_D = loss_D_real + loss_D_fake
        elif self.GAN_type == 'WGAN_GP' :
            beta = torch.rand(batch_size, 1, 1, 1).to(self.device)
            interpolates = beta * real.data + (1 - beta) * fake.data
            interpolates = interpolates.requires_grad_(True)
            interpolates_logits = self.D(interpolates, alpha)
            loss_D = - real_pred.mean() + fake_pred.mean() + 10 * self.GAN_Loss(interpolates_logits, interpolates).mean()

        self.D.zero_grad()
        loss_D.backward()
        self.optim_D.step()

        #########################################################################################
        fake = self.G(latent_vector, alpha)
        G_pred = self.D(fake,alpha)
        if self.GAN_type == 'LSGAN':
            loss_G = self.GAN_Loss(G_pred, real_label)
        elif self.GAN_type == 'WGAN_GP':
            loss_G = -G_pred.mean()
        self.G.zero_grad()
        loss_G.backward()
        self.optim_G.step()
        losses={}
        losses['real_pred'] = real_pred
        losses['fake_pred'] = fake_pred
        losses['loss_D'] = loss_D
        losses['loss_G'] = loss_G
        return losses, fake

    def save(self, opt, stage, iteration, device):
        path = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.isdir(path) :
            os.makedirs(path,exist_ok=True)
        if hasattr(self.G,'module') :
            multi = True
            self.single_gpu()
        else :
            multi = False
        torch.save({
            'iteration' : iteration,
            'G' : self.G.cpu().state_dict(),
            'D' : self.D.cpu().state_dict(),
            'optim_G' : self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
        }, os.path.join(path,f'model_{stage}_{iteration}'))
        self.G.to(device)
        self.D.to(device)
        if multi :
            self.multi_gpu()

    def load(self, opt):
        path = os.path.join(opt.load_dir)
        file = torch.load(path,map_location=self.device)
        if hasattr(self.G,'module') :
            multi = True
            self.single_gpu()
        else :
            multi = False
        self.G.load_state_dict(file['G'],strict=False)
        self.D.load_state_dict(file['D'],strict=False)
        self.optim_G.load_state_dict(file['optim_G'])
        self.optim_D.load_state_dict(file['optim_D'])
        if multi :
            self.multi_gpu()

class GPLoss(nn.Module):
    def __init__(self,device):
        super(GPLoss, self).__init__()
        self.device = device

    def forward(self,y,x):
        # y : interpolate logits
        # x : interpolate images
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx_l2norm = torch.norm(dydx.view(dydx.size(0), -1), 2, dim=1)
        loss = torch.mean((dydx_l2norm - 1) ** 2)
        return loss