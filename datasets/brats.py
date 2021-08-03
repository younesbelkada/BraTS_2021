import pydicom
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from glob import glob
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import csv
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
import torchvision.datasets as v_datasets
from datasets.brats_dataset import BraTS_Dataset_mean
from PIL import Image
from utils.utils_train import *
import logging
from sklearn.model_selection import train_test_split
class BraTS_mean_Dataloader():
    """
    Creates dataloader for train and val or test depending on the mode
    """
    def __init__(self, config):
        self.config     = config
        self.logger     = logging.getLogger("Cifar10DataLoader")

        transform = v_transforms.Compose(
                [v_transforms.CenterCrop(64),
                 v_transforms.ToTensor(),
                 v_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        if config.data_mode == "raw":
            pass

        elif config.data_mode == "images":
            self.logger.info(f"BraTS_mean_Dataloader, data_mode : {config.data_mode}, path : {self.config.img_dir}")
            dataset = BraTS_Dataset_mean(self.config.img_dir,self.config.annotation_file,transform, self.config.img_size)

            train_indices, valid_indices, _, _ = train_test_split(range(len(dataset)),dataset.targets,stratify=dataset.targets,test_size=self.config.valid_size)

            train_dataset = Subset(dataset, train_indices)
            valid_dataset = Subset(dataset, valid_indices)

            self.len_train_data = len(train_dataset)
            self.len_valid_data = len(valid_dataset)

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size


            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    

