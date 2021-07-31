import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 0.0001
epochs = 100
path_model = 'output/'