import torchvision
from dataset import get_data_loaders
import torch
import torch.nn as nn

from model import CNN_Model
from alex_net_model import AlexNet

from engine import train_model

def get_classes(data_dir):
    all_data = torchvision.datasets.ImageFolder(data_dir)
    return all_data.classes


DATASET_PATH = "CUB_200_2011/images"
(train_loader, train_data_len) = get_data_loaders(DATASET_PATH, 256, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(DATASET_PATH, 64, train=False)
classes = get_classes(DATASET_PATH)

dataloaders = {
    "train":train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = CNN_Model(len(classes))
model = AlexNet(len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
criterion = criterion.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, step_scheduler, device, num_epochs=11)

