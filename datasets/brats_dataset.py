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
    def __init__(self, img_dir, annotation_file, transform=None, build_=False, patches=True,split="train"):
        
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.ext = '*.dcm'
        self.tools = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        self.path_output = img_dir.replace('data','archives_mean')
        
        self.split = split

        self.transform = transform
        self.build_ = build_
        self.patches = patches
        self.im_size = 225
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
                mean_im = np.zeros((self.im_size,self.im_size), np.float)
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

            # this oprobably slows it down cuz of the accesses to label? Also every "self adds a reading to the data, should be local to each job"
            pt = (os.path.join(output_directory_subject, subjects+'_{}.png'.format(tool)))
            return [pt,label]


    def preprocess(self):
        # for now, I get black images.... but if you remove this, should work
        self.csv_df = pd.read_csv(self.annotation_file,header=0,index_col="BraTS21ID")

        array_data = os.listdir(os.path.join(self.img_dir, self.split))
        self.output_directory = os.path.join(self.path_output, self.split)
        if self.build_: 
            os.makedirs(self.output_directory,exist_ok=True)
        temp_data = (p_map(self.job,array_data,self.csv_df["MGMT_value"]))
        self.image_path,self.targets = zip(*temp_data)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        label = self.targets[idx]
        

        image = np.zeros((4,self.im_size,self.im_size), np.uint8)
        image[0, :, :] = np.array(Image.open(img_path).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[1, :, :] = np.array(Image.open(img_path.replace('FLAIR','T1w'  )).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[2, :, :] = np.array(Image.open(img_path.replace('FLAIR','T1wCE')).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))
        image[3, :, :] = np.array(Image.open(img_path.replace('FLAIR','T2w'  )).resize((self.im_size,self.im_size), Image.ANTIALIAS).convert('L'))

        input_tensor = torch.from_numpy(image).type(torch.float)
        if self.transform:
            input_tensor = self.transform(input_tensor)

        if self.patches:
            patches = input_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
            input_tensor = patches.contiguous().view(-1, 9, self.patch_size, self.stride)#.squeeze(0)

        return input_tensor, label
                


