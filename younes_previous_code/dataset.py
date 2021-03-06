# import pydicom
# import torch
# import numpy as np
# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from glob import glob
# import csv
# from torchvision import transforms
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# import itertools
# from utils.utils_train import *

# class BraTS_Dataset(Dataset):
#     def __init__(self, split, path, path_csv, tool_name=None, val_pr=0.01, transform=None):
#         self.split_ = split
#         if split == 'val':
#             split = 'train'
#         self.split = split
#         self.path = path
#         self.tool = tool_name
#         self.path_csv = path_csv
#         self.ext = '*.dcm'
#         self.val_pr = val_pr
#         self.preprocess()
#         self.transform = tranform

#     def preprocess(self):
#         self.path_to_images = []
#         self.image_labels = []
#         self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

#         array_data = os.listdir(os.path.join(self.path, self.split))
        
#         for subjects in array_data:
#             if self.tool:
#                 path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
#                 self.path_to_images.extend(path_to_images)
#                 # label = self.csv_df.loc[self.csv_df['BraTS21ID'] == subjects]
#                 label = self.csv_df.loc[int(subjects)].MGMT_value
#                 tab_labels = [label for i in range(len(path_to_images))]
#                 self.image_labels.extend(tab_labels)
#         N = int((1-self.val_pr)*len(self.image_labels))
#         if self.split_ == 'train':
#             self.image_labels = self.image_labels[:N]
#             self.path_to_images = self.path_to_images[:N]
#         elif self.split_ == 'val':
#             self.image_labels = self.image_labels[N:]
#             self.path_to_images = self.path_to_images[N:]

#     def __len__(self):
#         return len(self.image_labels)

#     def __getitem__(self, idx):
#         img_path = self.path_to_images[idx]
#         image = read_image(img_path)
#         label = self.image_labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # for each individual, create a tensor of size 500 which corresponds to 500 feature vectors

# class BraTS_Dataset_video(Dataset):
#     def __init__(self, split, path, path_csv, tool_name=None, val_pr=0.01, N_vec=500, transform=None, device=None, mode='sum'):
#         self.split_ = split
#         if split == 'val':
#             split = 'train'
#         self.split = split
#         self.device = device
#         self.path = path
#         self.tool = tool_name
#         self.path_csv = path_csv
#         self.ext = '*.dcm'
#         self.val_pr = val_pr
#         self.N_vec = N_vec
#         self.alexnet = models.alexnet(pretrained=True)
#         feat_extractor = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])
#         self.alexnet.classifier = feat_extractor
#         self.alexnet.eval()
#         self.alexnet = self.alexnet.to(self.device)
#         self.mode = mode
        
#         self.preprocess()
#         self.transform = transform

#     def preprocess(self):
#         self.dict_paths = {}
#         self.dict_labels = {}
#         self.csv_df = pd.read_csv(self.path_csv,header=0,index_col="BraTS21ID")

#         array_data = os.listdir(os.path.join(self.path, self.split))
        
#         for i, subjects in enumerate(array_data):
#             if self.tool:
#                 path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
#                 self.dict_paths[i] = path_to_images
#                 label = self.csv_df.loc[int(subjects)].MGMT_value
#                 self.dict_labels[i] = label
#                 #print(len(path_to_images))
#         N = int((1-self.val_pr)*len(self.dict_labels))
#         if self.split_ == 'train':
#             self.dict_labels = dict(list(self.dict_labels.items())[:N])
#             self.dict_paths = dict(list(self.dict_paths.items())[:N])
#         elif self.split_ == 'val':
#             self.dict_labels = dict(list(self.dict_labels.items())[N:])
#             self.dict_paths = dict(list(self.dict_paths.items())[N:])

#     def __len__(self):
#         return len(self.dict_labels)

#     def __getitem__(self, idx):
#         img_path = self.dict_paths[idx]
#         i = 0
#         array_tensors = []
#         if self.mode == 'sum':
#             input_tensor = torch.empty((1, 4096)).type(torch.float)
#             while i < self.N_vec:
#                 path = img_path[i]   
#                 image = read_image(path)
#                 if self.transform:
#                     image = self.transform(image.astype(np.uint8)).repeat(3, 1, 1).to(self.device)
#                 features = self.alexnet(image.unsqueeze(0))
#                 input_tensor += features
#                 i += 1
#             input_tensor /= i         
#         else:
#             input_tensor = torch.empty((self.N_vec, 4096)).type(torch.float)
#             while i < self.N_vec:
#                 if i < len(img_path):
#                     path = img_path[i]   
#                     image = read_image(path)
#                     if self.transform:
#                         image = self.transform(image.astype(np.uint8)).repeat(3, 1, 1).to(self.device)
#                     #print(image.dtype)
#                     features = self.alexnet(image.unsqueeze(0))
#                     #print(features.shape)
#                     array_tensors.append(features.detach().cpu())
#                     i += 1
#                     #input_tensor = torch.cat((input_tensor.detach().cpu(), features.detach().cpu()), dim=0)
#                 else:
#                     array_tensors.append(torch.zeros(1,4096))
#                     #input_tensor = torch.cat((input_tensor.detach().cpu(), torch.zeros(1,4096)), dim=0)
#                     i += 1
#             input_tensor = torch.cat(array_tensors, dim=0)
#         label = self.dict_labels[idx]
#         return input_tensor, label
