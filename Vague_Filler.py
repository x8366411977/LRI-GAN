from torch import nn
import torch
from utils import GatedConv2d,TransposeGatedConv2d


class Filler(nn.Module):
    def __init__(self):
        super(Filler, self).__init__()
        # Initialize the padding scheme
        self.block1 = nn.Sequential(
            # encoder
            GatedConv2d(2, 32, 5, 2, 2, activation='elu', norm='none', sc=True),
            GatedConv2d(32, 32, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(32, 64, 3, 2, 1, activation='elu', norm='none', sc=True)
        )
        self.block2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True)
        )
        self.block3 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True)
        )
        self.block4 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation='elu', norm='none', sc=True)
        )
        self.block5 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation='elu', norm='none', sc=True)
        )
        self.block6 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation='elu', norm='none', sc=True)
        )
        self.block7 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, activation='elu', norm='none', sc=True)
        )
        self.block8 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
        )
        # decoder
        self.block9 = nn.Sequential(
            TransposeGatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none', sc=True),
            TransposeGatedConv2d(64, 32, 3, 1, 1, activation='elu', norm='none', sc=True),
            GatedConv2d(32, 1, 3, 1, 1, activation='none', norm='none', sc=True),
            nn.Tanh()
        )

    def forward(self, first_in):
        first_out = self.block1(first_in)
        first_out = self.block2(first_out) + first_out
        first_out = self.block3(first_out) + first_out
        first_out = self.block4(first_out) + first_out
        first_out = self.block5(first_out) + first_out
        first_out = self.block6(first_out) + first_out
        first_out = self.block7(first_out) + first_out
        first_out = self.block8(first_out) + first_out
        first_out = self.block9(first_out)
        first_out = torch.clamp(first_out, 0, 1)
        return first_out