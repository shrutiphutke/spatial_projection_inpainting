from os import listdir
from os.path import join
import random

from PIL import Image
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder_Test(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder_Test, self).__init__()
        self.a_path = join(image_dir, "input")
        # self.b_path = join(image_dir, "target")
        self.c_path = join(image_dir, "mask")

        self.path = image_dir
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        # print(join(self.a_path, self.image_filenames[index]))
        # print(join(self.b_path, self.image_filenames[index]))
        # print(join(self.c_path, self.image_filenames[index]))

        a = cv2.imread(join(self.a_path, self.image_filenames[index]))
        # b = cv2.imread(join(self.b_path, self.image_filenames[index]))
        c = cv2.imread(join(self.c_path, self.image_filenames[index]))

        a = cv2.resize(a, (256, 256),  interpolation = cv2.INTER_CUBIC)
        # b = cv2.resize(b, (256, 256),  interpolation = cv2.INTER_CUBIC)
        c = cv2.resize(c, (256, 256),  interpolation = cv2.INTER_CUBIC)

        a = transforms.ToTensor()(a)
        # b = transforms.ToTensor()(b)
        c = transforms.ToTensor()(c)

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        c = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(c)

        return a, c, self.image_filenames[index]
        

    def __len__(self):
        return len(self.image_filenames)