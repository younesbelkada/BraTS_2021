from utils.dataset import *
# create a config file 
path_dataset = './data'
path_csv = './data/train_labels.csv'
tool_name = 'T2w'

test_dataset = BraTS_Dataset('train', path_dataset, path_csv, tool_name)
