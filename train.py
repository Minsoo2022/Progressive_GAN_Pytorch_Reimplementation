import torch
import argparse
import torchvision
import numpy as np
import tensorboardX
import os
from models import Progressive_GAN
from dataloader import get_loader
from config import get_train_config
from utils import *


def get_train_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', help="name")
    parser.add_argument("--data_dir", type=str, default='../../dataset/FFHQ', help="location of dataset")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="location of checkpoint")
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard', help="location of tensorboard")
    parser.add_argument("--init_stage", type=int, default=0, help="init stage")
    parser.add_argument("--last_stage", type=int, default=8, help="last stage")
    parser.add_argument("--G_lr", type=float, default=0.001, help="learning rate for the generator")
    parser.add_argument("--D_lr", type=float, default=0.001, help="learning rate for the discriminator")
    parser.add_argument("--gpu_ids", type=str, default='0', help="GPU ids")

    return parser.parse_args()

def train(opt, config, model, board) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_Latent = torch.randn(8, config['ch_Latent']).to(device)
    for stage in range(opt.init_stage,opt.last_stage+1) :
        batch_size = config['batch_size_list'][stage]
        max_iter = config['max_iter_list'][stage]
        #num = 0
        dataloader = iter(get_loader(opt,stage,batch_size))
        for i in range(max_iter):
            #num += batch_size
            try :
                real_image = next(dataloader)
            except StopIteration:
                dataloader = (iter(get_loader(opt, stage, batch_size)))
                real_image = next(dataloader)
            alpha = min(i / (max_iter//2), 1)
            print(stage, i, alpha)
            losses, images = model.train(real_image, alpha)
            if i % 1000 == 0 :
                with torch.no_grad() :
                    val_images = model.G(val_Latent, alpha)
                add_losses(board, losses, stage, i)
                add_images(board, images, stage, i, 'Train')
                add_images(board, val_images, stage, i, 'Validation')
            if i % 20000 == 0:
                model.save(opt, i, device)
        model.save(opt, max_iter, device)
        model.stage_up()


if __name__=="__main__" :
    opt = get_train_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    config = get_train_config()
    model = Progressive_GAN(opt, config)
    board = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    train(opt, config, model, board)


# def train(opt, config, model, board) :
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     val_Latent = torch.randn(8, config['ch_Latent']).to(device)
#     for stage in range(opt.init_stage,opt.last_stage+1) :
#         batch_size = config['batch_size_list'][stage]
#         max_iter = config['max_iter_list'][stage]
#         iteration = 0
#         stop = False
#         dataloader = iter(get_loader(opt,stage,batch_size))
#         for i in range(max_iter//batch_size + 1):
#             iteration += batch_size
#             try :
#                 real_image = next(dataloader)
#             except StopIteration:
#                 dataloader = (iter(get_loader(opt, stage, batch_size)))
#                 real_image = next(dataloader)
#             if iteration >= max_iter :
#                 stop = True
#                 real_image = real_image[:batch_size-(iteration-max_iter)]
#                 iteration = max_iter
#             alpha = min(iteration / (max_iter//2), 1)
#             print(iteration, alpha)
#             losses, images = model.train(real_image, alpha)
#             if iteration % 1000 == 0 :
#                 with torch.no_grad() :
#                     val_images = model.G(val_Latent, alpha=1)
#                 add_losses(board, losses, stage, iteration)
#                 add_images(board, images, stage, iteration, 'Train')
#                 add_images(board, val_images, stage, iteration, 'Validation')
#             if iteration % 20000 == 0:
#                 model.save(opt.checkpoint_dir, iteration)
#             if stop :
#                 continue
#         model.save(opt.checkpoint_dir, max_iter)
#         model.stage_up()