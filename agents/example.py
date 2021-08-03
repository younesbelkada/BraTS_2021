import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here

from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from utils.utils_train import binary_acc, print_summary_step

cudnn.benchmark = True


class ExampleAgent(BaseAgent):

    def __init__(self, config, model, data_loader, loss, optimizer):
        super().__init__(config)

        # define models
        self.model = model

        # define data_loader
        self.data_loader = data_loader

        # define loss
        self.loss = loss

        # define optimizers for both generator and discriminator
        self.optimizer = optimizer

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
           self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        #self.cuda = self.is_cuda & self.config.cuda
        self.cuda = self.is_cuda

        # set the manual seed for torch
        #self.manual_seed = self.config.seed
        if self.cuda:
            #torch.cuda.manual_seed_all(self.manual_seed)
            #torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self, epochs):
        """
        Main training loop
        :return:
        """

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        pass

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
