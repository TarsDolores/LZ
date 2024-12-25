# -*- coding:utf-8 -*-
# @Time : 2023/5/23 15:30
# @Author : Lei Li

import os
import torch.utils.data as data
import numpy as np
from PIL import Image


class Remote(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')
        self.split = split

        if self.split in ['train', 'val']:
            self.images_dir = os.path.join(self.root, self.split, 'images')
            self.labels_dir = os.path.join(self.root, self.split, 'labels')
            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')
            self.images = []
            self.labels = []

            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))

                target_name = file_name.replace('tif', 'png')
                self.labels.append(os.path.join(self.labels_dir, target_name))
        else:
            self.images_dir = os.path.join(self.root, self.split, 'images')
            if not os.path.isdir(self.images_dir):
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')
            self.images = []

            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))


    def __getitem__(self, index):
        if self.split in ['train', 'val']:

            image = Image.open(self.images[index]).convert('RGB')
            image = np.array(image)
            target = Image.open(self.labels[index])
            target = np.array(target)

            if self.transform:
                transformed = self.transform(image=image, mask=target)
                image, target = transformed['image'], transformed['mask']
            target = np.array(target)

            file_name = os.path.basename(self.images[index]).split('.')[0]

            return image, target, file_name
        else:
            image = Image.open(self.images[index]).convert('RGB')
            image = np.array(image)

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            file_name = os.path.basename(self.images[index]).split('.')[0]
            return image, file_name

    def __len__(self):
        return len(self.images)
