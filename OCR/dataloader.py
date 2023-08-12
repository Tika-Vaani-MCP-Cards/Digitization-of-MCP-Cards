import os
import cv2
import six
import lmdb
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transforms import *
from utils import *


class MCPDatesDataset(Dataset):
    """Dates Dataset"""

    def __init__(self, root_dir, opt=None, is_training=True, data_aug=True, img_w=144, img_h=32):
        """
        Args:
            root_dir (string): Directory with lmdb file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_w=img_w
        self.img_h=img_h
        self.opt = opt
        self.root_dir = root_dir
        self.is_training = is_training
        self.data_aug = data_aug
        self.env = lmdb.open(self.root_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        self.filtered_index_list = [] # keep indices of valid samples only

        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVRandomCrop(px_max=20, p = 0.6),
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.6),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

        # load index_to_Imgname dict
        with open(os.path.join(self.root_dir, 'pickleDict/index_to_imgName.pkl'), 'rb') as f:
            self.index_to_fileName = pickle.load(f)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            for index in range(nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                # check date validity
                if is_valid_date(label):
                    self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)
            print('Total samples- ', self.nSamples)

    def __len__(self):
        return self.nSamples
        
    def resize(self, img):
        return cv2.resize(img, (self.img_w, self.img_h))

    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))  # TODO:move is_training to here

    def __getitem__(self, idx):
        index = self.filtered_index_list[idx]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            img_abs_path = self.index_to_fileName[img_key.decode()]
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

        # cv2.imwrite('original.jpg', np.array(img))
        input_channel = 3 if img.mode == 'RGB' else 1

        if self.is_training:
            img = self._process_training(img)
        else:
            img = self._process_test(img)
        # print('transformed/{}_{}.jpg'.format(idx,label))
        # cv2.imwrite('transformed/{}_{}.jpg'.format(idx,label.replace('/','-')), img)

        img = torch.tensor(img).float()
        if input_channel == 1:
            img = img.unsqueeze(-1)

        img = torch.permute(img, (2, 0, 1))

        return img, label, img_abs_path

