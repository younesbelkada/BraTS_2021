import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent
from graphs.models.mlp_mixer import MLP_Mixer
from graphs.losses.example import BinaryCrossEntropy
from datasets.brats import BraTS_Dataset_mean

# import your classes here

from utils.metrics import AverageMeter, AverageMeterList, cls_accuracy
from utils.misc import print_cuda_statistics
from utils.utils_train import binary_acc, print_summary_step
from utils.train_utils import adjust_learning_rate

cudnn.benchmark = True


class MLP_MixerAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = MLP_Mixer(self.config)

        # define data_loader
        self.data_loader = BraTS_Dataset_mean(self.config)

        # define loss
        self.loss = BinaryCrossEntropy()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config.learning_rate,
                                         momentum=float(self.config.momentum),
                                         weight_decay=self.config.weight_decay,
                                         nesterov=True)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
           self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        
        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self, epochs):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)
        


    def train_one_epoch(self):
        """
        One epoch of training
        uses tqdm to load data in parallel
        :return:
        """
        tqdm_batch = tqdm(self.data_loader, total=len(self.data_loader),
                          desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters @TODO: ARTHUR I still don't know what that is
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # current iteration over total iterations
            progress = float(self.current_epoch * self.data_loader.train_iterations + current_batch) / (
                    self.config.max_epoch * self.data_loader.train_iterations)
            # progress = float(self.current_iteration) / (self.config.max_epoch * self.data_loader.train_iterations)
            x, y = Variable(x), Variable(y)
            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.train_iterations)
            # model
            pred = self.model(x, progress)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))

            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1.item(), x.size(0))
            top5_acc.update(top5.item(), x.size(0))

            self.current_iteration += 1
            current_batch += 1

            self.summary_writer.add_scalar("epoch/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", top1_acc.val, self.current_iteration)
        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader, total=self.data_loader,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1.item(), x.size(0))
            top5_acc.update(top5.item(), x.size(0))

        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.avg) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
