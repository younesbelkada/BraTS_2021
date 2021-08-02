from dataset.dataset import *
from graphs.model.mixer import *
from agents.example import ExampleAgent

path_dataset = './archive'
path_csv = './archive/train_labels.csv'
tool_name = 'FLAIR'

train_dataset = BraTS_Dataset_volume('train', path_dataset, path_csv, tool_name)
val_dataset = BraTS_Dataset_volume('val', path_dataset, path_csv, tool_name)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

model = MLP_Mixer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.BCELoss()

agent = ExampleAgent(None, model, train_dataloader, criterion, optimizer)
agent.train(10)

