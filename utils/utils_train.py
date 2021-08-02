from sklearn.metrics import average_precision_score
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import torch

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.flatten(y_pred))
    y_test = torch.flatten(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc * 100
    #print(acc)
    return acc

def print_summary_step(step, train_loss, acc):
    text = "Step : {} | ".format(step) + "m_step_loss : %.3f | m_step_acc : %.1f " % (train_loss, acc)
    print('{}'.format(text), end="\r")

def average_precision(output, target):
    return average_precision_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())