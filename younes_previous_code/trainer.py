import numpy as np
from math import floor
import torch
import cv2
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def print_summary_step(step, train_loss, acc):
    text = "Step : {} | ".format(step) + "m_step_loss : %.3f | m_step_acc : %.1f " % (train_loss, acc)
    print('{}'.format(text), end="\r")

class Trainer():
    """
        Class definition for training and saving the trained model
    """
    def __init__(self, model, criterion, train_data, val_data, optimizer, epochs, device, path_model):
        self.model = model
        self.criterion = criterion
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.path_model = path_model
    def train(self):
        running_loss = 0
        i = 0
        best_ap = 0
        best_ac = 0
        grads_x = []
        grads = []
        self.model.train()
        
        for epoch in range(self.epochs):
            #self.parser.model = self.parser.model.train().to(self.parser.device)
            losses = []
            accuracies = []

            for x_batch, y_batch in train_loader:
                
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch.float())
                running_loss += loss.item()
                
                loss.backward()
                losses.append(loss.item())
                accuracies.append(binary_acc(output.type(torch.float), y_batch).item())

                self.optimizer.step()
                i += 1

                if i%10 == 0:
                    print_summary_step(i, np.mean(losses), np.mean(accuracies))
                    losses = []
                    accuracies = []

            i = 0
            #best_ap, best_ac, ap_val, acc_val = self.eval_epoch(best_ap, best_ac)
            print('')
            print('Epoch {} | mAP_val : {} | mAcc_val :{}'.format(epoch+1, ap_val, acc_val))



    def eval_epoch(self, best_ap, best_ac):
        self.model.eval()
        


        if aps > best_ap:
            best_ap = aps
            best_ac = accs
            torch.save(self.model.state_dict(), self.path_model)
        #self.parser.model = self.parser.model.train().to(self.parser.device)
        return best_ap, best_ac, aps, accs