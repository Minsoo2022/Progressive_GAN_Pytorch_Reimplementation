from utils import *

import torch
from torch import nn
import numpy as np
import math

Upsampling = nn.Upsample(scale_factor=2,mode='nearest')
Downsampling = nn.Upsample(scale_factor=1/2,mode='bilinear')
class Generator(nn.Module):
    def __init__(self, stage = 0, ch_Latent=512, depth_list=[], norm='PixelNorm', equalized=True):
        super(Generator,self).__init__()
        self.stage = 0
        self.ch_Latent = ch_Latent
        self.norm = norm
        self.equalized = equalized
        self.main_layers = nn.ModuleList()
        self.to_RGB_layers = nn.ModuleList()
        self.depth_list = depth_list
        self.layers_init()
        if stage > 0 :
            for i in range(stage) :
                self.stage_up()

    def layers_init(self):
        self.main_layers.append(G_Linear(self.ch_Latent, self.ch_Latent, norm=self.norm, equalized=self.equalized))
        self.main_layers.append(G_ConvBlock(self.ch_Latent, self.depth_list[0], first_layer=True, norm=self.norm, equalized=self.equalized))
        self.to_RGB_layers.append(Equalized_layer(nn.Conv2d(self.depth_list[0], 3, 1, 1, 0), self.equalized))

    def stage_up(self):
        self.stage += 1
        self.main_layers.append(G_ConvBlock(self.depth_list[self.stage-1], self.depth_list[self.stage], norm=self.norm, equalized=self.equalized))
        self.to_RGB_layers.append(Equalized_layer(nn.Conv2d(self.depth_list[self.stage],3,1,1,0),self.equalized))

    def forward(self, x, alpha):
        for i, layer in enumerate(self.main_layers[:-1]):
            x = layer(x)
            if layer._get_name() == 'G_Linear':
                x= x.reshape(-1, self.ch_Latent, 1, 1)
            if layer._get_name() == 'G_ConvBlock':
                x = Upsampling(x)
        if alpha != 1 and self.stage != 0:
            y = self.to_RGB_layers[-2](x)
        x = self.to_RGB_layers[-1](self.main_layers[-1](x))
        if alpha != 1 and self.stage != 0:
            x = x * alpha + y * (1 - alpha)
        return x

class G_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LReLU', norm='PixelNorm', first_layer=False, equalized=True):
        super(G_ConvBlock, self).__init__()
        self.first_layer = first_layer
        self.Activation = activation
        self.Conv_1 = Equalized_layer(nn.Conv2d(in_channels,out_channels,kernel_size=4 if first_layer else 3,stride=1,padding=3 if first_layer else 1, bias= True), equalized)
        self.Conv_2 = Equalized_layer(nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1, bias=True),equalized)
        if activation == 'LReLU' :
            self.Activation = nn.LeakyReLU(0.2)
        if norm == 'InstanceNorm' :
            self.Norm = nn.InstanceNorm2d(out_channels)
        elif norm =='PixelNorm' :
            self.Norm = Pixel_Norm()

    def forward(self, x):
        x = self.Norm(self.Activation(self.Conv_1(x)))
        x = self.Norm(self.Activation(self.Conv_2(x)))
        return x

class G_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LReLU', norm='PixelNorm', equalized=True):
        super(G_Linear, self).__init__()
        self.Activation = activation
        self.Linear = Equalized_layer(nn.Linear(in_channels, out_channels), equalized)
        if activation == 'LReLU' :
            self.Activation = nn.LeakyReLU(0.2)
        if norm == 'InstanceNorm' :
            self.Norm = nn.InstanceNorm2d(out_channels)
        elif norm =='PixelNorm' :
            self.Norm = Pixel_Norm()

    def forward(self, x):
        x = self.Norm(self.Activation(self.Linear(x)))
        return x

class Discriminator(nn.Module):
    def __init__(self, stage = 0, ch_Latent=512, depth_list=[], norm='PixelNorm', equalized=True):
        super(Discriminator,self).__init__()
        self.stage = 0
        self.ch_Latent = ch_Latent
        self.norm = norm
        self.equalized = equalized
        self.main_layers = nn.ModuleList()
        self.from_RGB_layers = nn.ModuleList()
        self.depth_list = depth_list
        self.layers_init()
        if stage > 0 :
            for i in range(stage) :
                self.stage_up()

    def layers_init(self):
        self.main_layers.append(Equalized_layer(nn.Linear(self.ch_Latent, 1), self.equalized))
        self.main_layers.append(D_ConvBlock(self.depth_list[0]+1, self.ch_Latent, first_layer=True, equalized=self.equalized))
        self.from_RGB_layers.append(Equalized_layer(nn.Conv2d(3, self.depth_list[0], 1, 1, 0), self.equalized))

    def stage_up(self):
        self.stage += 1
        self.main_layers.append(D_ConvBlock(self.depth_list[self.stage], self.depth_list[self.stage-1]))
        self.from_RGB_layers.append(nn.Conv2d(3,self.depth_list[self.stage],1,1,0))

    def get_minibatch_standard_deviation(self, x):
        size = x.shape
        y = x.std([0]).mean().expand(size[0],1,size[2],size[3])
        return y

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
        MSD = self.get_minibatch_standard_deviation(x)
        x = torch.cat((x,MSD), dim=1)
        x = self.main_layers[1](x)
        x = x.reshape(-1, self.ch_Latent)
        x = self.main_layers[0](x)
        return x

class D_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LReLU', first_layer=False, equalized=True):
        super(D_ConvBlock, self).__init__()
        self.first_layer = first_layer
        self.Conv_1 = Equalized_layer(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), equalized)
        self.Conv_2 = Equalized_layer(nn.Conv2d(in_channels, out_channels, kernel_size=4 if first_layer else 3, stride=1, padding=0 if first_layer else 1, bias=True), equalized)
        if activation == 'LReLU':
            self.Activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.Activation(self.Conv_1(x))
        if self.first_layer :
            return self.Conv_2(x)
        x = self.Activation(self.Conv_2(x))
        return x


class Equalized_layer(nn.Module):
    def __init__(self, layer, equalized):
        super(Equalized_layer,self).__init__()
        self.layer = layer
        self.equalized = equalized
        if self.equalized :
            self.layer.weight.data.normal_(0,1)  # init mean 0 std 1 normal distribution
            self.layer.bias.data.fill_(0) # init bias 0
            self.weight = self.cal_He_const()

    def forward(self, x):
        if self.equalized :
            return self.layer(x) * self.weight
        else :
            return self.layer(x)

    def cal_He_const(self):
        size = self.layer.weight.size()
        fan_in = np.prod(size[1:])
        return math.sqrt(2.0 / fan_in)

class Pixel_Norm():
    def __call__(self, x):
        epsilon = 1e-8
        square_mean = x.pow(2).mean(1) + epsilon
        return x/square_mean.sqrt().unsqueeze(1)