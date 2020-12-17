import torch
import argparse
import torchvision
import numpy as np
from models import Progressive_GAN
from dataloader import get_loader

def get_train_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../../dataset/FFHQ', help="location of dataset")
    parser.add_argument("--init_stage", type=int, default=0, help="init stage")
    parser.add_argument("--last_stage", type=int, default=8, help="init stage")
    return parser.parse_args()

def train(opt, model) :
    for stage in range(opt.init_stage,opt.last_stage+1) :
        dataloader = get_loader(opt,stage)
        for i, real_image in enumerate(dataloader):
            alpha = 1
            print(i)
            model.train(real_image,alpha)
        model.stage_up()

if __name__=="__main__" :
    opt = get_train_parse()
    model = Progressive_GAN(opt.init_stage)
    train(opt, model)


