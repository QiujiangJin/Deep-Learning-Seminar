import os, glob
import numpy as np
import torch as tr
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json
import skimage
from skimage.io import imread
from skimage import data

class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, test=False):
        self.root_dir = root_dir
        self.mode = 'test' if test else 'train'
        self.csv_file = csv_file
        self.load_csv()
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop((210, 320)),
                transforms.ToTensor()
            ]
        )

    def load_csv(self):
        img_path_df = pd.read_csv(self.csv_file, header=None)
        self.img_list = list(img_path_df.iloc[:, 0])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir+'Images/'+self.img_list[idx]).convert('RGB')
        truth = Image.open(self.root_dir+'GroundTruth/'+self.img_list[idx].split('.')[0]+'_GT.bmp').convert('RGB')
        if image.size[1] == 320:
            image = image.rotate(90, expand=True)
            truth = truth.rotate(90, expand=True)
        mask = toInt(np.asarray(truth))
        
        image = self.transform(image)
        truth = self.transform(truth)
        mask = self.transform(mask).type(tr.LongTensor)
        return image, truth, mask[0]

def toInt(img):
    #mask = - np.ones(img.shape[:2], dtype=int)
    mask = np.zeros(img.shape[:2], dtype=int)
    color_map = json.load(open('../data/color_map.json'))
    for line in color_map:
        idx = np.all(img == line['rgb_values'], axis=2)
        #mask[idx] = line['id']
        mask[idx] = line['id']+1
    mask = mask*1.0
    return Image.fromarray(mask)

def toRGB(img):
    rgb_template = tr.zeros(img.shape[0], img.shape[1], img.shape[2], 3)
    color_map = json.load(open('../data/'+'color_map.json'))
    for line in color_map:
        #rgb_template[img==line['id']] = tr.tensor(line['rgb_values']).type(tr.FloatTensor)
        rgb_template[img==(line['id']+1)] = tr.tensor(line['rgb_values']).type(tr.FloatTensor)
    label = (rgb_template/255.0).permute(0, 3, 1, 2)
    return label

if __name__ == '__main__':
    dataset = ImageDataset(
        'data/MSRC_ObjCategImageDatabase_v2/', 'data/TextonBoostSplits/Test.txt', test=False)
    for d in dataset[0]:
        print(d.shape)
