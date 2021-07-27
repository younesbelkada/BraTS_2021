from utils.dataset import *
# create a config file 
path_dataset = './data'
path_csv = './data/train_labels.csv'
tool_name = 'T2w'

test_dataset = BraTS_Dataset('train', path_dataset, path_csv, tool_name)
mi,ma=1000,-1
for img, label in test_dataset:
    mi = min(img.shape[0],mi)
    ma = max(img.shape[0],ma)
    print(np.min(img),np.max(img))
print(mi,ma)
