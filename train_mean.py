from dataset.dataset import *
from graphs.model.mixer import *
from agents.example import ExampleAgent

path_dataset = './archive'
path_csv = './archive/train_labels.csv'
tool_name = 'FLAIR'
patches = True
build = False
batch_size = 4
epochs = 100

transform = transforms.Compose([
    transforms.RandomRotation((-180, 180), fill=0)
])

train_dataset_mean = BraTS_Dataset_mean('train', path_dataset, path_csv, 0.01, transform, build, patches)
val_dataset_mean = BraTS_Dataset_mean('val', path_dataset, path_csv, 0.01, transform, build, patches)

train_dataloader = DataLoader(train_dataset_mean, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset_mean, batch_size=batch_size, shuffle=True)

model = MLP_Mixer(in_channels=4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

criterion = nn.BCELoss()

agent = ExampleAgent(None, model, train_dataloader, criterion, optimizer)
agent.train(epochs,val_dataloader)
