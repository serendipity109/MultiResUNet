import torch
import torch.nn as nn
import torch.nn.functional as F
from .Conv2d_modified import Conv2d_samepadding

class conv2d_bn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation='relu'):
        super().__init__()
        self.conv2d = Conv2d_samepadding(in_channels, out_channels, kernel_size, padding='SAME')
        self.bn = nn.BatchNorm2d(out_channels, affine=False, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        
        if(self.activation == None):
            return x
        elif self.activation == 'relu':
            relu = nn.ReLU(inplace=True)
            x = relu(x)
            return x
        elif self.activation == 'sigmoid':
            sigmoid = nn.Sigmoid()
            x = sigmoid(x)
            return x           

# def conv2d_bn(x, in_channels, out_channels, kernel_size, activation='relu'):
#     conv2d = Conv2d_samepadding(in_channels, out_channels, kernel_size, padding='SAME')
#     x = conv2d(x)
#     bn = nn.BatchNorm2d(out_channels)
#     x = bn(x)
#     if(activation == None):
#         return x
#     elif activation == 'relu':
#         relu = nn.ReLU(inplace=True)
#         x = relu(x)
#         return x
#     elif activation == 'sigmoid':
#         relu = nn.Sigmoid(inplace=True)
#         x = relu(x)
#         return x        

class MultiResBlock(nn.Module):

    def __init__(self, in_channels, U, alpha = 1.67):
        super().__init__()
        self.in_channels = in_channels
        W = alpha * U
        self.bn = nn.BatchNorm2d(int(W*0.167) + int(W*0.333) + \
                             int(W*0.5))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = conv2d_bn(self.in_channels, (int(W*0.167) + int(W*0.333) + \
                             int(W*0.5)), 1, activation=None)
        self.conv3 = conv2d_bn(self.in_channels, int(W*0.167), 3, activation='relu')
        self.conv5 = conv2d_bn(int(W*0.167), int(W*0.333), 3, activation='relu')
        self.conv7 = conv2d_bn(int(W*0.333), int(W*0.5), 3, activation='relu')

    def forward(self, x):
        shortcut = self.shortcut(x)
        conv3x3 = self.conv3(x)
        conv5x5 = self.conv5(conv3x3)
        conv7x7 = self.conv7(conv5x5)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim = 1)

        out = self.bn(out)
        out = out + shortcut
        out = self.relu(out)
        out = self.bn(out)
        return out

class ResPath(nn.Module):

    def __init__(self, in_channels, out_channels, length):
        super().__init__()    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.s0 = conv2d_bn(self.in_channels, out_channels, 1, activation=None)
        self.o0 = conv2d_bn(self.in_channels, out_channels, 3, activation='relu')

        self.length = length
        self.sn = conv2d_bn(self.out_channels, out_channels, 1, activation=None)
        self.on = conv2d_bn(self.out_channels, out_channels, 3, activation='relu')

    def forward(self, x):
        shortcut = self.s0(x)
        out = self.o0(x)
        out = out + shortcut
        out = self.relu(out)
        out = self.bn(out)        

        for i in range(self.length-1):
            shortcut = self.sn(out)
            out = self.on(out)
            out = out + shortcut
            out = self.relu(out)
            out = self.bn(out)

        return out

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return x
