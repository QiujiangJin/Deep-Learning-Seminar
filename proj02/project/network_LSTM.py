import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_LSTM import ConvLSTM

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTM(64, 256, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTM(256, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTM(512, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)

    def forward(self, input, h_1, h_2, h_3):
        
        h_1 = self.rnn1(self.conv(input), h_1)
        h_2 = self.rnn2(h_1[0], h_2)
        h_3 = self.rnn3(h_2[0], h_3)

        return h_3[0], h_1, h_2, h_3


class Binarizer(nn.Module):
    def __init__(self, bottleneck):
        super(Binarizer, self).__init__()
        
        self.bottleneck = bottleneck
        self.conv = nn.Conv2d(512, self.bottleneck, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        
        return torch.tanh(self.conv(input)).sign()


class Decoder(nn.Module):
    def __init__(self, bottleneck):
        super(Decoder, self).__init__()

        self.bottleneck = bottleneck
        self.conv1 = nn.Conv2d(self.bottleneck, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTM(512, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTM(128, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTM(128, 256, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3, bias=False)
        self.rnn4 = ConvLSTM(64, 128, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, h_1, h_2, h_3, h_4):
        
        h_1 = self.rnn1(self.conv1(input), h_1)
        h_2 = self.rnn2(F.pixel_shuffle(h_1[0], 2), h_2)
        h_3 = self.rnn3(F.pixel_shuffle(h_2[0], 2), h_3)
        h_4 = self.rnn4(F.pixel_shuffle(h_3[0], 2), h_4)

        return torch.tanh(self.conv2(F.pixel_shuffle(h_4[0], 2))), h_1, h_2, h_3, h_4