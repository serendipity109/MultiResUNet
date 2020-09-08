from .unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MultiResUnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.mresblock1 = MultiResBlock(in_channels, 32)
        self.respath1 = ResPath(51, 32, 4)

        self.mresblock2 = MultiResBlock(51, 32*2)
        self.respath2 = ResPath(105, 32*2, 3)    

        self.mresblock3 = MultiResBlock(105, 32*4)
        self.respath3 = ResPath(212, 32*4, 2)              

        self.mresblock4 = MultiResBlock(212, 32*8)
        self.respath4 = ResPath(426, 32*8, 1)   

        self.mresblock5 = MultiResBlock(426, 32*16)   

        self.up6 = Up(853, 32*8)
        self.mresblock6 = MultiResBlock(32*16, 32*8)

        self.up7 = Up(426, 32*4)
        self.mresblock7 = MultiResBlock(32*8, 32*4)

        self.up8 = Up(212, 32*2)
        self.mresblock8 = MultiResBlock(32*4, 32*2)

        self.up9 = Up(105, 32)
        self.mresblock9 = MultiResBlock(32*2, 32)        

        self.conv10 = conv2d_bn(51, 1, 1, activation='sigmoid')
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        m1 = self.mresblock1(x)
        p1 = self.pool(m1)
        m1 = self.respath1(m1)

        m2 = self.mresblock2(p1)
        p2 = self.pool(m2)
        m2 = self.respath2(m2)

        m3 = self.mresblock3(p2)
        p3 = self.pool(m3)
        m3 = self.respath3(m3)

        m4 = self.mresblock4(p3)
        p4 = self.pool(m4)
        m4 = self.respath4(m4)

        m5 = self.mresblock5(p4)

        u6 = self.up6(m5, m4)
        m6 = self.mresblock6(u6)

        u7 = self.up7(m6, m3)
        m7 = self.mresblock7(u7)    

        u8 = self.up8(m7, m2)
        m8 = self.mresblock8(u8)                

        u9 = self.up9(m8, m1)
        m9 = self.mresblock9(u9) 

        x = self.conv10(m9)
        return x

# def MultiResUnet(height, width, n_channels):

#     inputs = Input((height, width, n_channels))

#     mresblock1 = MultiResBlock(32, inputs)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
#     mresblock1 = ResPath(32, 4, mresblock1)

#     mresblock2 = MultiResBlock(32*2, pool1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
#     mresblock2 = ResPath(32*2, 3, mresblock2)

#     mresblock3 = MultiResBlock(32*4, pool2)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
#     mresblock3 = ResPath(32*4, 2, mresblock3)

#     mresblock4 = MultiResBlock(32*8, pool3)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
#     mresblock4 = ResPath(32*8, 1, mresblock4)

#     mresblock5 = MultiResBlock(32*16, pool4)

#     up6 = concatenate([Conv2DTranspose(
#         32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
#     mresblock6 = MultiResBlock(32*8, up6)

#     up7 = concatenate([Conv2DTranspose(
#         32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
#     mresblock7 = MultiResBlock(32*4, up7)

#     up8 = concatenate([Conv2DTranspose(
#         32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
#     mresblock8 = MultiResBlock(32*2, up8)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
#         2, 2), padding='same')(mresblock8), mresblock1], axis=3)
#     mresblock9 = MultiResBlock(32, up9)

#     conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
#     model = Model(inputs=[inputs], outputs=[conv10])

#     return model