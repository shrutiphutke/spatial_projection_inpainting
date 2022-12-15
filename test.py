from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import get_test_set

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset_path', default='dataset/', required=False, help='facades')
parser.add_argument('--save_path', default='outputs', required=False, help='facades')
parser.add_argument('--checkpoints_path', default='checkpoints', required=False, help='facades')
parser.add_argument('--dataset_name', default='paris', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', action='store_false', help='use cuda')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_path+'_'+opt.dataset_name+'/'):
        os.makedirs(opt.save_path+'_'+opt.dataset_name+'/')

device = torch.device("cuda:0" if opt.cuda else "cpu")


G_path = opt.checkpoints_path+'/'+opt.dataset_name+'/'+"netG.pth"
my_net = torch.load(G_path).to(device)  



test_set = get_test_set(opt.dataset_path+opt.dataset_name+'/')
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


for iteration_test, batch in enumerate(testing_data_loader,1):
    input, mask_test, filename = batch[0].to(device), batch[1].to(device), batch[2]


    prediction, prediction2 = my_net(input, mask_test)
    
    prediction_img2 = prediction2.detach().squeeze(0).cpu()
    save_img(prediction_img2, "./{}/{}".format(opt.save_path+'_'+opt.dataset_name+'/',filename[0]))


