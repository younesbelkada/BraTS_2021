from utils.dataset import *
# create a config file 
path_dataset = ''
path_csv = ''
tool_name = 'FLAIR'
test_dataset = BraTS_Dataset('train', path_dataset, path_csvm, tool_name)

for img, label in test_dataset:
    print(img.shape)
    print(label)
    exit(0)