import os
import sys
import argparse
import torch
#torch.cuda.set_device(1)
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from models import *
from dataloader import *
from train import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=4, help='batch size')
parser.add_argument(
    '--epochs', '-e', type=int, default=1000, help='max epochs')
parser.add_argument(
    "--model_name", type=str, default="Unet_plus-1", help="Name of model")
parser.add_argument(
    '--patience', type=int, default=20, help='early stopping')
args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
    print('cuda is available!!')
else:
    args.device = torch.device('cpu')
    print('no cuda')
    assert False
sys.stdout.flush()

print(args)
sys.stdout.flush()

train_data = ImageDataset(root_dir='../data/MSRC_ObjCategImageDatabase_v2/', csv_file='../data/TextonBoostSplits/Train.txt')
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_data = ImageDataset(root_dir='../data/MSRC_ObjCategImageDatabase_v2/', csv_file='../data/TextonBoostSplits/Validation.txt')
#val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)

print("!start training!")
sys.stdout.flush()
train(args, train_dataloader, val_dataloader)

