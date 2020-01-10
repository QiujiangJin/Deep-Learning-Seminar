import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.block_1 = BasicBlock(3, 64, 9, 4)
        self.block_2 = BasicBlock(64, 128, 3, 1)
        self.block_3 = BasicBlock(128, 256, 3, 1)
        self.block_4 = BasicBlock(256, 256, 3, 1)
        self.block_5 = BasicBlock(256, 128, 9, 4)
        self.conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 3, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(3)
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        return x

class PerceptualLoss(nn.Module):
    
    def __init__(self, vgg):
        super().__init__()
        self.vgg_features = vgg.features
        self.layers = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        
    def forward(self, x):
        outputs = dict()
        for name, module in self.vgg_features._modules.items():
            x = module(x)
            if name in self.layers:
                outputs[self.layers[name]] = x
        return outputs