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
import pydicom
from p_tqdm import p_map

class BraTS_Dataset_mean(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None, build_=False, patches=False,split="train"):
        
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.ext = '*.dcm'
        self.tools = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        self.path_output = 'archives_mean'
        
        self.split = split

        self.transform = transform
        self.build_ = build_
        self.patches = patches
        self.im_size = 224
        self.patch_size = 75
        self.stride = 75
        self.image_path = []
        self.targets = []
        self.preprocess()

    def job(self,subjects,label):
        output_directory_subject = os.path.join(self.output_directory, subjects)
        os.makedirs(output_directory_subject,exist_ok=True)
        for tool in self.tools:
            path_to_images = glob(os.path.join(self.img_dir, self.split, subjects, tool, self.ext))
            if self.build_:
                mean_im = np.zeros((224,224), np.float)
                for im in path_to_images:
                    # depending on the file extension, either get the np array, or process the image directly 
                    if self.ext == "*.dcm":
                        dicom = pydicom.read_file(im)
                        data = dicom.pixel_array
                        if np.min(data)==np.max(data): data = np.zeros((self.im_size,self.im_size))
                        data = data - np.min(data)
                        if np.max(data) != 0: data = data / np.max(data)
                        mean_im +=  cv2.resize(data, (self.im_size,self.im_size))
                    elif self.ext == "*.png":
                        pil_im = np.array(Image.open(im).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
                        mean_im += pil_im
                    else:
                        raise Exception
                if len(path_to_images) != 0:
                    mean_im = (mean_im/len(path_to_images)).astype(np.uint8)
                mean_pil_im = Image.fromarray(mean_im.astype(np.uint8))
                mean_pil_im.save(os.path.join(output_directory_subject, subjects+'_{}.png'.format(tool)))
            self.targets.append(label)
            self.image_path.append(os.path.join(output_directory_subject, subjects+'_{}.png'.format(tool)))


    def preprocess(self):
        self.dict_paths = {}
        self.dict_labels = {}
        self.dict_subject = {}
        self.csv_df = pd.read_csv(self.annotation_file,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.img_dir, self.split))

        if self.build_: 
            self.output_directory = os.path.join('archives_mean', self.split)
            os.makedirs(self.output_directory,exist_ok=True)

        print("starting _job")
        # process in parallel
        print(p_map(self.job,array_data,self.csv_df["MGMT_value"]))

    def __len__(self):
        return len(self.dict_labels)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        subject = img_path.split("/")[-1][:-4]

        print(subject)
        exit(0)

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
                


