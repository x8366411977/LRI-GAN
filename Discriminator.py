import torch
from utils import Conv2dLayer
from torch import nn

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.sn = True
        self.norm = 'in'
        self.block1 = Conv2dLayer(2, 64, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block2 = Conv2dLayer(64, 128, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block3 = Conv2dLayer(128, 256, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block4 = Conv2dLayer(256, 256, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block5 = Conv2dLayer(256, 256, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block6 = Conv2dLayer(256, 16, 3, 2, 1, activation='lrelu', norm=self.norm, sn=self.sn)
        self.block7 = torch.nn.Linear(256, 1)

    def forward(self, img, mask):

        x = torch.cat((img, mask), 1)
        x = self.block1(x)  # out: [B, 64, 256, 256]
        x = self.block2(x)  # out: [B, 128, 128, 128]
        x = self.block3(x)  # out: [B, 256, 64, 64]
        x = self.block4(x)  # out: [B, 256, 32, 32]
        x = self.block5(x)  # out: [B, 256, 16, 16]
        x = self.block6(x)  # out: [B, 256, 8, 8]
        x = x.reshape([x.shape[0], -1])
        x = self.block7(x)
        return x