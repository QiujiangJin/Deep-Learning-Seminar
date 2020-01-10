import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=0, dilation=1, hidden_kernel_size=1, bias=False):
        
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.hidden_kernel_size = hidden_kernel_size
        self.hidden_stride = 1
        self.hidden_padding = hidden_kernel_size//2
        self.hidden_dilation = 1
        self.gate_channels = 4*self.hidden_channels
        
        self.conv_xh = nn.Conv2d(in_channels=self.input_channels, out_channels=self.gate_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=bias)
        
        self.conv_hh = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.gate_channels, kernel_size=self.hidden_kernel_size, stride=self.hidden_stride, padding=self.hidden_padding, dilation=self.hidden_dilation, bias=bias)

    def forward(self, input, hidden):
        
        h_cur, c_cur = hidden
        g = self.conv_xh(input) + self.conv_hh(h_cur)
        i, f, c, o = g.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)
        c_new = (f*c_cur) + (i*c)
        h_new = o*torch.tanh(c_new)

        return h_new, c_new