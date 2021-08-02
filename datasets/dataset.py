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

class BraTS_Dataset(Dataset):
    def __init__(self, split, path, path_csv, tool_name=None, val_pr=0.01, transform=None):
        self.split_ = split
        if split == 'val':
            split = 'train'
        self.split = split
        self.path = path
        self.tool = tool_name
        self.path_csv = path_csv
        self.ext = '*.dcm'
        self.val_pr = val_pr
        self.preprocess()
        self.transform = tranform

    def preprocess(self):
        self.path_to_images = []
        self.image_labels = []
        self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.path, self.split))
        
        for subjects in array_data:
            if self.tool:
                path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
                self.path_to_images.extend(path_to_images)
                # label = self.csv_df.loc[self.csv_df['BraTS21ID'] == subjects]
                label = self.csv_df.loc[int(subjects)].MGMT_value
                tab_labels = [label for i in range(len(path_to_images))]
                self.image_labels.extend(tab_labels)
        N = int((1-self.val_pr)*len(self.image_labels))
        if self.split_ == 'train':
            self.image_labels = self.image_labels[:N]
            self.path_to_images = self.path_to_images[:N]
        elif self.split_ == 'val':
            self.image_labels = self.image_labels[N:]
            self.path_to_images = self.path_to_images[N:]

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = self.path_to_images[idx]
        image = read_image(img_path)
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# for each individual, create a tensor of size 500 which corresponds to 500 feature vectors

class BraTS_Dataset_video(Dataset):
    def __init__(self, split, path, path_csv, tool_name=None, val_pr=0.01, N_vec=500, transform=None, device=None, mode='sum'):
        self.split_ = split
        if split == 'val':
            split = 'train'
        self.split = split
        self.device = device
        self.path = path
        self.tool = tool_name
        self.path_csv = path_csv
        self.ext = '*.dcm'
        self.val_pr = val_pr
        self.N_vec = N_vec
        self.alexnet = models.alexnet(pretrained=True)
        feat_extractor = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])
        self.alexnet.classifier = feat_extractor
        self.alexnet.eval()
        self.alexnet = self.alexnet.to(self.device)
        self.mode = mode
        
        self.preprocess()
        self.transform = transform

    def preprocess(self):
        self.dict_paths = {}
        self.dict_labels = {}
        self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.path, self.split))
        
        for i, subjects in enumerate(array_data):
            if self.tool:
                path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
                self.dict_paths[i] = path_to_images
                label = self.csv_df.loc[int(subjects)].MGMT_value
                self.dict_labels[i] = label
                #print(len(path_to_images))
        N = int((1-self.val_pr)*len(self.dict_labels))
        if self.split_ == 'train':
            self.dict_labels = dict(list(self.dict_labels.items())[:N])
            self.dict_paths = dict(list(self.dict_paths.items())[:N])
        elif self.split_ == 'val':
            self.dict_labels = dict(list(self.dict_labels.items())[N:])
            self.dict_paths = dict(list(self.dict_paths.items())[N:])

    def __len__(self):
        return len(self.dict_labels)

    def __getitem__(self, idx):
        img_path = self.dict_paths[idx]
        i = 0
        array_tensors = []
        if self.mode == 'sum':
            input_tensor = torch.empty((1, 4096)).type(torch.float)
            while i < self.N_vec:
                path = img_path[i]   
                image = read_image(path)
                if self.transform:
                    image = self.transform(image.astype(np.uint8)).repeat(3, 1, 1).to(self.device)
                features = self.alexnet(image.unsqueeze(0))
                input_tensor += features
                i += 1
            input_tensor /= i         
        else:
            input_tensor = torch.empty((self.N_vec, 4096)).type(torch.float)
            while i < self.N_vec:
                if i < len(img_path):
                    path = img_path[i]   
                    image = read_image(path)
                    if self.transform:
                        image = self.transform(image.astype(np.uint8)).repeat(3, 1, 1).to(self.device)
                    #print(image.dtype)
                    features = self.alexnet(image.unsqueeze(0))
                    #print(features.shape)
                    array_tensors.append(features.detach().cpu())
                    i += 1
                    #input_tensor = torch.cat((input_tensor.detach().cpu(), features.detach().cpu()), dim=0)
                else:
                    array_tensors.append(torch.zeros(1,4096))
                    #input_tensor = torch.cat((input_tensor.detach().cpu(), torch.zeros(1,4096)), dim=0)
                    i += 1
            input_tensor = torch.cat(array_tensors, dim=0)
        label = self.dict_labels[idx]
        return input_tensor, label

class BraTS_Dataset_volume(Dataset):
    def __init__(self, split, path, path_csv, tool_name=None, val_pr=0.01, transform=None, N_vec=500):
        self.split_ = split
        if split == 'val':
            split = 'train'
        self.split = split
        self.path = path
        self.tool = tool_name
        self.path_csv = path_csv
        self.ext = '*.png'
        self.val_pr = val_pr
        self.N_vec = N_vec
        self.preprocess()
        self.stride=75
        self.patch_size=75
        self.transform = transform

    def preprocess(self):
        self.dict_paths = {}
        self.dict_labels = {}
        self.dict_subject = {}
        self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.path, self.split))
        N = int((1-self.val_pr)*len(array_data))
        if self.split_ == 'train':
            array_data = array_data[:N]
        if self.split_ == 'val':
            array_data = array_data[N:]
        for i, subjects in enumerate(array_data):
            if self.tool:
                path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
                self.dict_paths[i] = path_to_images
                if self.split != 'test':
                    label = self.csv_df.loc[int(subjects)].MGMT_value
                else:
                    label = None
                self.dict_labels[int(subjects)] = label
                self.dict_subject[i] = int(subjects)

    def __len__(self):
        return len(self.dict_labels)

    def __getitem__(self, idx):
        img_path = self.dict_paths[idx]
        i = 0
        array_tensors = []
        sorted_img_path = {}
        for img in img_path:
            nb_frame = int(img.split('/')[-1][:-4].split('-')[-1])
            sorted_img_path[nb_frame] = img
        sorted_img_path = dict(sorted(sorted_img_path.items())) 

        input_tensor = torch.zeros((9, self.N_vec, 75, 75)).type(torch.float)
        offset = int((self.N_vec-len(img_path))/2)
        while i < self.N_vec:
            if i < len(img_path):
                path = list(sorted_img_path.values())[i]
                image = Image.open(path).resize((225,225), Image.ANTIALIAS).convert('L')
                image = transforms.ToTensor()(image)
                #stride = 75
                patches = image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
                patches = patches.contiguous().view(-1, 9, 75, 75).squeeze(0)
                for j in range(9):
                    patch = patches[j, :, :]
                    input_tensor[j, i+offset, :, :] = patch
            i += 1
        true_index = self.dict_subject[idx]
        label = self.dict_labels[true_index]
        return input_tensor, label

class BraTS_Dataset_mean(Dataset):
    def __init__(self, split, path, path_csv, val_pr=0.01, transform=None, buid_=False, patches=False):
        self.split_ = split
        if split == 'val':
            split = 'train'
        self.split = split
        self.path = path
        self.path_csv = path_csv
        self.ext = '*.png'
        self.val_pr = val_pr
        self.tools = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        self.path_output = 'archives_mean'
        
        self.transform = transform
        self.buid_ = buid_
        self.patches = patches
        self.im_size = 225
        self.patch_size = 75
        self.stride = 75
        
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
                


