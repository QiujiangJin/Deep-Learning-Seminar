import numpy as np
import sys
import os
import torch
from .models import UNet
from torchvision import datasets, transforms

def Trans(img):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    return transform(img)

def segment(img):
    """
    Semantically segment an image
    img: an uint8 numpy of size (w,h,3)
    return: a numpy integer array of size (w,h), where the each entry represent the class id
    please refer to data/color_map.json for the id <-> class mapping
    """
    sample_img = Trans(img)
    sample_img = torch.unsqueeze(sample_img, 0)
    sample_img = sample_img.to(device)
    output = MODEL(sample_img)
    softmax=torch.argmax(output, dim=1)
    new_img = softmax[0].cpu().numpy()
    return (new_img-1)
    #return (new_img)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available!!')
else:
    device = torch.device('cpu')
    print('no cuda')

sys.stdout.flush()
MODEL = UNet(22).to(device)
#MODEL = UNet(21).to(device)
MODEL.load_state_dict(torch.load('./project/save/Unet_plus-1'))
#MODEL.load_state_dict(torch.load('./project/save/Unet'))
MODEL.eval()
