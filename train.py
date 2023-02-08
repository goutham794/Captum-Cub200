import torchvision
from dataset import get_data_loaders
import torch
import torch.nn as nn
import wandb

from cnn_model import CNN_Model
import config
from engine import train_model

def get_classes(data_dir):
    all_data = torchvision.datasets.ImageFolder(data_dir)
    return all_data.classes


def run(wandb_proj_name):

    wandb.login(key=config.wandb_api_key)
    wandb.init(
    project=wandb_proj_name,
    config={
        "epochs": 50,
        "batch_size": 256,
        "lr": 1e-5,
        "dropout": 0.1,
        })

    wandb_config = wandb.config


    DATASET_PATH = "CUB_200_2011/images"
    (train_loader, train_data_len) = get_data_loaders(DATASET_PATH, wandb_config['batch_size'], train=True)
    (val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(DATASET_PATH, wandb_config['batch_size'], train=False)
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

    model = AlexNet(len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.96)

    model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, step_scheduler, device, num_epochs = wandb_config['epochs'])

    torch.save(model_ft, 'alex_net.pt')

if __name__ == "__main__":
    wandb_proj_name = "captum-cnn-label_smooth_0.1_lr_0.001"
    run(wandb_proj_name)
