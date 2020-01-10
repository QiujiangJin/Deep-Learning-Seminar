import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(In, Out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Out),
            nn.ReLU(),
            nn.Conv2d(Out, Out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Out),
            nn.ReLU()
        )

    def forward(self, x):
    	out = self.double_conv(x)

    	return out

class T_Double(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.up = nn.ConvTranspose2d(In, Out, kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(In, Out)

    def forward(self, x, y):
        y = self.up(y)
        d_1 = x.size()[3] - y.size()[3]
        d_2 = x.size()[2] - y.size()[2]
        y = F.pad(y, [d_1//2, d_1 - d_1//2, d_2//2, d_2 - d_2//2])
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)

        return out

class M_Double(nn.Module):
    def __init__(self, In, Out):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(In, Out)

    def forward(self, x):
        out = self.conv(self.maxpool(x))
        
        return out

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes

        self.In = DoubleConv(3, 64)
        self.down1 = M_Double(64, 128)
        self.down2 = M_Double(128, 256)
        self.down3 = M_Double(256, 512)
        self.down4 = M_Double(512, 1024)
        
        self.up1 = T_Double(1024, 512)
        self.up2 = T_Double(512, 256)
        self.up3 = T_Double(256, 128)
        self.up4 = T_Double(128, 64)
        self.Out = nn.Conv2d(64, self.n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        tmp_1 = self.In(x)
        tmp_2 = self.down1(tmp_1)
        tmp_3 = self.down2(tmp_2)
        tmp_4 = self.down3(tmp_3)
        tmp_5 = self.down4(tmp_4)

        out = self.up1(tmp_4, tmp_5)
        out = self.up2(tmp_3, out)
        out = self.up3(tmp_2, out)
        out = self.up4(tmp_1, out)
        out = self.Out(out)

        return out