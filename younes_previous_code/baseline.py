from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import config as c

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
criterion = nn.CrossEntropyLoss()

train_dataset = BraTS_Dataset('train', path_dataset, path_csv, tool_name)
train_dataloader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)

trainer = Trainer(model, criterion, train_dataloader, None, optimizer, c.epochs, c.device, c.path_model)