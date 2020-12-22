from utils import *

import torch
from torch import nn

Upsampling = nn.Upsample(scale_factor=2,mode='nearest')
Downsampling = nn.Upsample(scale_factor=1/2,mode='bilinear')
class Generator(nn.Module):
    def __init__(self, stage = 0, ch_Latent=512, depth_list=[]):
        super(Generator,self).__init__()
        self.stage = 0
        self.ch_Latent = ch_Latent
        self.main_layers = nn.ModuleList()
        self.to_RGB_layers = nn.ModuleList()
        self.depth_list = depth_list
        self.main_layers.append(nn.Linear(self.ch_Latent, 16 * self.ch_Latent))
        self.main_layers.append(G_ConvBlock(self.ch_Latent, self.depth_list[0], first_layer=True))
        self.to_RGB_layers.append(nn.Conv2d(self.depth_list[0],3,1,1,0))
        if stage > 0 :
            for i in range(stage) :
                self.stage_up()

    def stage_up(self):
        self.stage += 1
        self.main_layers.append(G_ConvBlock(self.depth_list[self.stage-1], self.depth_list[self.stage]))
        self.to_RGB_layers.append(nn.Conv2d(self.depth_list[self.stage],3,1,1,0))

    def forward(self, x, alpha):
        for i, layer in enumerate(self.main_layers[:-1]):
            x = layer(x)
            if layer._get_name() == 'Linear':
                x= x.reshape(-1, self.ch_Latent, 4, 4)
            if layer._get_name() == 'G_ConvBlock':
                x = Upsampling(x)
        if alpha != 1 and self.stage != 0:
            y = self.to_RGB_layers[-2](x)
        x = self.to_RGB_layers[-1](self.main_layers[-1](x))
        if alpha != 1 and self.stage != 0:
            x = x * alpha + y * (1 - alpha)
        return x

class Discriminator(nn.Module):
    def __init__(self, stage = 0, ch_Latent=512, depth_list=[]):
        super(Discriminator,self).__init__()
        self.stage = 0
        self.ch_Latent = ch_Latent
        self.main_layers = nn.ModuleList()
        self.from_RGB_layers = nn.ModuleList()
        self.depth_list = depth_list
        self.main_layers.append(nn.Linear(ch_Latent, 1))
        self.main_layers.append(D_ConvBlock(self.depth_list[0], ch_Latent, first_layer=True))
        self.from_RGB_layers.append(nn.Conv2d(3,self.depth_list[0],1,1,0))
        if stage > 0 :
            for i in range(stage) :
                self.stage_up()

    def stage_up(self):
        self.stage += 1
        self.main_layers.append(D_ConvBlock(self.depth_list[self.stage], self.depth_list[self.stage-1]))
        self.from_RGB_layers.append(nn.Conv2d(3,self.depth_list[self.stage],1,1,0))

    def forward(self, x, alpha):
        if alpha != 1 and self.stage != 0:
            y = self.from_RGB_layers[-2](Downsampling(x))
        x = self.from_RGB_layers[-1](x)
        for i, layer in enumerate(reversed(self.main_layers[2:])):
            x = layer(x)
            if layer._get_name() == 'D_ConvBlock':
                x = Downsampling(x)
            if i == 0 and alpha != 1 and self.stage != 0:
                x = x * alpha + y * (1 - alpha)
        x = self.main_layers[1](x)
        x = x.reshape(-1, self.ch_Latent)
        x = self.main_layers[0](x)
        return x

class G_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LReLU', norm='InstanceNorm', first_layer=False):
        super(G_ConvBlock, self).__init__()
        self.first_layer = first_layer
        #Conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=4 if first_layer else 3,stride=1,padding=2 if first_layer else 1, bias= True)
        self.Conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=True)
        self.Conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1, bias= True)
        if activation == 'LReLU' :
            self.Activation = nn.LeakyReLU(0.2)
        if norm == 'InstanceNorm' :
            self.Norm = nn.InstanceNorm2d(out_channels)
    def forward(self, x):
        x = self.Norm(self.Activation(self.Conv_1(x)))
        x = self.Norm(self.Activation(self.Conv_2(x)))
        return x

class D_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LReLU', norm='InstanceNorm', first_layer=False):
        super(D_ConvBlock, self).__init__()
        self.first_layer = first_layer
        #Conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=4 if first_layer else 3,stride=1,padding=2 if first_layer else 1, bias= True)
        if first_layer :
            self.Conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            self.Conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=True)
        else :
            self.Conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1, bias=True)
            self.Conv_2 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1, bias= True)
        if activation == 'LReLU':
            self.Activation = nn.LeakyReLU(0.2)
        if norm == 'InstanceNorm' :
            self.Norm = nn.InstanceNorm2d(out_channels)
    def forward(self, x):
        x = self.Norm(self.Activation(self.Conv_1(x)))
        if self.first_layer :
            return self.Conv_2(x)
        x = self.Norm(self.Activation(self.Conv_2(x)))
        return x

# print(1)
# G = Generator(0)
# D = Discriminator(0)
# r = torch.rand(512).unsqueeze(0)
# D(G(r,1),1)
# print(2)
# G.stage_up()
# D.stage_up()
# D(G(r,1),1)
# print(3)
# G.stage_up()
# D.stage_up()
# D(G(r,1),1)
# print(3)