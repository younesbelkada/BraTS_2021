from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as v_transforms
from datasets.brats_dataset import BraTS_Dataset_mean
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
                [v_transforms.RandomRotation((-180, 180), fill=0)])

        if config.data_mode == "raw":
            pass

        elif config.data_mode == "images":
            self.logger.info(f"BraTS_mean_Dataloader, data_mode : {config.data_mode}, path : {self.config.img_dir}")
            dataset = BraTS_Dataset_mean(self.config.img_dir,self.config.annotation_file,transform,self.config.build,split="train")

            train_indices, valid_indices = train_test_split(range(len(dataset)),test_size=self.config.valid_size,train_size=1-self.config.valid_size,shuffle=False)

            train_dataset = Subset(dataset, train_indices)
            valid_dataset = Subset(dataset, valid_indices)

            self.len_train_data = len(train_dataset)
            self.len_valid_data = len(valid_dataset)

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size


            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    
    def finalize(self):
        pass
