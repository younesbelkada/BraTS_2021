import pydicom
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import csv
from torchvision import transforms
from pydicom.pixel_data_handlers.util import apply_voi_lut
import itertools
from PIL import Image
from utils.utils_train import *
import logging

class BraTS_Dataset_mean(Dataset):
    def __init__(self, config):
        self.config     = config
        self.logger     = logging.getLogger("Cifar10DataLoader")
    
        self.split      = [self.config.mode,"train"][config.mode == "val"]
        self.path       = config.data_folder
        self.path_csv   = config.path_csv
        self.ext        = '*.png'
        
        
        self.val_pr     = config.val_pr
        self.tools      = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        self.output     = 'archives_mean'
        
        self.transform  = self.config.transform
        self.buid_      = self.config.build
        self.patches    = self.config.patches
        self.im_size    = self.config.im_size
        self.patch_size = self.config.patch_size
        self.stride     = self.config.stride
        
        self.preprocess()
    
    def preprocess(self):
        self.dict_paths = {}
        self.dict_labels = {}
        self.dict_subject = {}
        self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.path, self.split))
        N = int((1-self.val_pr)*len(array_data))
        output_directory = os.path.join('archives_mean', self.split)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if self.split_ == 'train':
            array_data = array_data[:N]
        if self.split_ == 'val':
            array_data = array_data[N:]
        for i, subjects in tqdm(enumerate(array_data)):
            output_directory_subject = os.path.join(output_directory, subjects)
            if not os.path.exists(output_directory_subject):
                os.makedirs(output_directory_subject)
            for tool in self.tools:
                path_to_images = glob(os.path.join(self.path, self.split, subjects, tool, self.ext))
                if self.buid_:
                    mean_im = np.zeros((224,224), np.float)
                    for im in path_to_images:
                        pil_im = np.array(Image.open(im).resize((224,224), Image.ANTIALIAS).convert('L'))
                        mean_im += pil_im
                    if len(path_to_images) != 0:
                        mean_im = (mean_im/len(path_to_images)).astype(np.uint8)
                    mean_pil_im = Image.fromarray(mean_im.astype(np.uint8))
                    mean_pil_im.save(os.path.join(output_directory_subject, subjects+'_{}.png'.format(tool)))
            
            label = self.csv_df.loc[int(subjects)].MGMT_value
            self.dict_paths[i] = output_directory_subject
            self.dict_labels[int(subjects)] = label
            self.dict_subject[i] = int(subjects)

            self.train_iterations = (self.dict_paths*(1-self.val_pr) + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.dict_paths*self.val_pr + self.config.batch_size - 1) // self.config.batch_size


    def __len__(self):
        return len(self.dict_labels)

    def __getitem__(self, idx):
        img_path = self.dict_paths[idx]
        subject = str(self.dict_subject[idx]).zfill(5)
        image = np.zeros((4,self.im_size,self.im_size), np.uint8)
        image[0, :, :] = np.array(Image.open(os.path.join(img_path, subject+'_FLAIR.png')).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[1, :, :] = np.array(Image.open(os.path.join(img_path, subject+'_T1w.png')).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[2, :, :] = np.array(Image.open(os.path.join(img_path, subject+'_T1wCE.png')).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[3, :, :] = np.array(Image.open(os.path.join(img_path, subject+'_T2w.png')).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))

        input_tensor = torch.from_numpy(image)
        if self.transform:
            input_tensor = self.transform(input_tensor)

        true_index = self.dict_subject[idx]
        label = self.dict_labels[true_index]
        #print(input_tensor.shape)
        if self.patches:
            patches = input_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
            input_tensor = patches.contiguous().view(-1, 9, self.patch_size, self.stride)#.squeeze(0)
        #print(input_tensor.shape)
        #exit(0)
        return input_tensor, label
                


