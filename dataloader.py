import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

def get_loader(opt, stage, batch_size) :
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = FFHQ_dataset(transform=transform, opt=opt, stage=stage)
    dataloader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=opt.workers)
    return dataloader

class FFHQ_dataset(data.Dataset):
    def __init__(self, transform=None, opt=None, stage=0):
        super(FFHQ_dataset,self).__init__()
        self.transform = transform
        data_dir = opt.data_dir
        resolution = 2**(stage+2)
        self.image_path = os.path.join(data_dir, f'FFHQ_{resolution}')
        self.image_list = glob.glob(os.path.join(self.image_path,'*.png'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(self.image_list[item])
        return self.transform(image)