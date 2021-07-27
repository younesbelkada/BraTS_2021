import pydicom
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
import csv
from torchvision import transforms
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_image(path_img, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path_img)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    #data = (data * 255).astype(np.uint8)
        
    return data

class BraTS_Dataset(Dataset):
    def __init__(self, split, path, path_csv, tool_name=None):
        self.split = split
        self.path = path
        self.tool = tool_name
        self.path_csv = path_csv
        self.ext = '.dcm'
        self.preprocess()
    def preprocess(self):
        self.path_to_images = []
        self.image_labels = []
        self.csv_df = pd.read_csv(self.path_csv)


        array_data = os.listdir(os.path.join(self.path, self.split))
        
        for subjects in array_data:
            if self.tool_name:
                path_to_images = glob(os.path.join(self.path, self.split, subjects, self.tool, self.ext))
                self.path_to_images.extend(path_to_images)
                #label = self.csv_df[self.csv_df['BraTS21ID'] == subjects]
                label = self.csv_df['BraTS21ID'][subjects].MGTM_value
                tab_labels = [label for i in range(len(path_to_images))]
                self.image_labels.extend(tab_labels)
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.path_to_images[idx]
        image = read_image(img_path)
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label