from torch import nn
import torch
from utils import GatedConv2d,TransposeGatedConv2d
from torchvision import transforms
from torch.nn import functional as F
from Vague_Filler import Filler

class GatedGenerator(nn.Module):
    def __init__(self):
        super(GatedGenerator, self).__init__()

        ########################################## Vague-Filler Network ##################################################
        self.coarse = Filler()

        ########################################## Generator Network #########################################################
        self.refinement1 = nn.Sequential( # 高宽减半
            GatedConv2d(1, 32, 5, 2, 2, activation='elu', norm='none'),                 
            GatedConv2d(32, 32, 3, 1, 1, activation='elu', norm='none'),
        )
        self.refinement2 = nn.Sequential( # 高宽再减半
            # encoder
            GatedConv2d(32, 64, 3, 2, 1, activation='elu', norm='none'),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none')
        )
        self.refinement3 = nn.Sequential( # 高宽再减半
            GatedConv2d(64, 128, 3, 2, 1, activation='elu', norm='none')
        )
        self.refinement4 = nn.Sequential( 
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none'),
        )
        self.refinement5 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 2, dilation = 2, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 4, dilation = 4, activation='elu', norm='none')
        )
        self.refinement6 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 8, dilation = 8, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 16, dilation = 16, activation='elu', norm='none'),
        )
        self.Transpose_refinement1 = nn.Sequential( 
            GatedConv2d(1, 32, 5, 2, 2, activation='elu', norm='none'),            
            GatedConv2d(32, 32, 3, 1, 1, activation='elu', norm='none'),
        )
        self.Transpose_refinement2 = nn.Sequential(
            # encoder
            GatedConv2d(32, 64, 3, 2, 1, activation='elu', norm='none'),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none')
        )
        self.Transpose_refinement3 = nn.Sequential( 
            GatedConv2d(64, 128, 3, 2, 1, activation='elu', norm='none')
        )
        self.Transpose_refinement4 = nn.Sequential( 
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none'),
        )
        self.Transpose_refinement5 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 2, dilation = 2, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 4, dilation = 4, activation='elu', norm='none')
        )
        self.Transpose_refinement6 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 8, dilation = 8, activation='elu', norm='none'),
            GatedConv2d(128, 128, 3, 1, 16, dilation = 16, activation='elu', norm='none'),
        )
        self.refinement7 = nn.Sequential(
            GatedConv2d(512, 128, 3, 1, 1, activation='elu', norm='none'),
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none')
        )
        self.refinement8 = nn.Sequential(
            TransposeGatedConv2d(192, 64, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(64, 32, 3, 1, 1, activation='elu', norm='none')
        )
        self.refinement9 = nn.Sequential(
            TransposeGatedConv2d(96, 32, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(32, 1, 3, 1, 1, activation='none', norm='none'),
            nn.Tanh()
        )
        self.conv_pl3 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none')
        )
        self.conv_pl2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation='elu', norm='none')
        )
        self.conv_pl1 = nn.Sequential(
            GatedConv2d(32, 32, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(32, 32, 3, 1, 2, dilation=2, activation='elu', norm='none')
        )
        self.conv_Transpose_pl3 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation='elu', norm='none')
        )
        self.conv_Transpose_pl2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation='elu', norm='none')
        )
        self.conv_Transpose_pl1 = nn.Sequential(
            GatedConv2d(32, 32, 3, 1, 1, activation='elu', norm='none'),
            GatedConv2d(32, 32, 3, 1, 2, dilation=2, activation='elu', norm='none')
        )

    def forward(self, img, mask):
        img_128 = F.interpolate(img, size=[128, 128], mode='bilinear')
        mask_128 = F.interpolate(mask, size=[128, 128], mode='nearest')
        ori_img = img_128 * (1 - mask_128) + mask_128
        first_masked_img = img_128 * (1 - mask_128) + mask_128
        first_in = torch.cat((first_masked_img, mask_128), 1)
        first_out = self.coarse(first_in)
        first_out = F.interpolate(first_out, size=[256,256], mode='bilinear')
        # Refinement

        # first_out = img * (1 - mask)
        second_in = img * (1 - mask) + first_out * mask
        Transpose_second_in = transforms.RandomHorizontalFlip(p=1)(second_in)

        # 原始图像
        pl1 = self.refinement1(second_in)             
        pl2 = self.refinement2(pl1)                         
        second_out = self.refinement3(pl2)              
        second_out = self.refinement4(second_out) + second_out    
        second_out = self.refinement5(second_out) + second_out
        pl3 = self.refinement6(second_out) +second_out   
        
        #Calculate Transpose_Attention
        patch_fb = self.cal_patch(16, mask, 256)
        att = self.compute_attention(pl3, patch_fb)

        Transpose_pl1 = self.Transpose_refinement1(Transpose_second_in)              
        Transpose_pl2 = self.Transpose_refinement2(Transpose_pl1)                     
        Transpose_second_out = self.Transpose_refinement3(Transpose_pl2)               
        Transpose_second_out = self.Transpose_refinement4(Transpose_second_out) + Transpose_second_out  
        Transpose_second_out = self.Transpose_refinement5(Transpose_second_out) + Transpose_second_out
        Transpose_pl3 = self.Transpose_refinement6(Transpose_second_out) +Transpose_second_out          

        # Calculate Attention
        patch_fb = self.cal_patch(16, mask, 256)
        Transpose_att = self.compute_attention(Transpose_pl3, patch_fb)

        second_out = torch.cat((pl3,self.conv_pl3(self.attention_transfer(pl3, att)),Transpose_pl3,self.conv_Transpose_pl3(self.attention_transfer(Transpose_pl3, Transpose_att))), 1) #out: [B, 512, 64, 64]
        second_out = self.refinement7(second_out)                                                 #out: [B, 64, 128, 128]
        second_out = torch.cat((second_out,self.conv_pl2(self.attention_transfer(pl2, att)),self.conv_Transpose_pl2(self.attention_transfer(Transpose_pl2, Transpose_att))), 1) #out: [B, 192, 128, 128]
        second_out = self.refinement8(second_out)                                                 #out: [B, 32, 256, 256]
        second_out = torch.cat((second_out,self.conv_pl1(self.attention_transfer(pl1, att)),self.conv_Transpose_pl1(self.attention_transfer(Transpose_pl1, Transpose_att))), 1) #out: [B, 96, 256, 256]
        second_out = self.refinement9(second_out)
        second_out = torch.clamp(second_out, 0, 1)
        return first_out, second_out
    
    def cal_patch(self, patch_num, mask, raw_size):
        pool = nn.MaxPool2d(raw_size // patch_num) 
        patch_fb = pool(mask) 
        return patch_fb

    def compute_attention(self, feature, patch_fb): 
        b = feature.shape[0]
        feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')
        p_fb = torch.reshape(patch_fb, [b, 16 * 16, 1])
        p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
        f = feature.permute([0, 2, 3, 1]).reshape([b, 16 * 16, 128])
        c = self.cosine_Matrix(f, f) * p_matrix
        s = F.softmax(c, dim=2) * p_matrix
        return s

    def attention_transfer(self, feature, attention):  
        b_num, c, h, w = feature.shape
        f = self.extract_image_patches(feature, 16)
        f = torch.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
        f = torch.bmm(attention, f)
        f = torch.reshape(f, [b_num, 16, 16, h // 16, w // 16, c])
        f = f.permute([0, 5, 1, 3, 2, 4])
        f = torch.reshape(f, [b_num, c, h, w])
        return f

    def extract_image_patches(self, img, patch_num):
        b, c, h, w = img.shape
        img = torch.reshape(img, [b, c, patch_num, h//patch_num, patch_num, w//patch_num])
        img = img.permute([0, 2, 4, 3, 5, 1])
        return img
    

    def cosine_Matrix(self, _matrixA, _matrixB):
        _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
        _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
        _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
        return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))