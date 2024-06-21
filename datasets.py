import torch
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

class InpaintDataset(Dataset):
    def __init__(self, path,image_szie = 256):
        self.path = path
        self.imglist = []
        for root, dirs, files in os.walk(self.path):
            for filespath in files:
                self.imglist.append(os.path.join(root, filespath))
        self.image_szie = image_szie

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index], cv2.IMREAD_GRAYSCALE)
        if self.imglist[index]:
            # try:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # except:
            #     print(self.imglist[index])
            img = cv2.resize(img, (self.image_szie,self.image_szie))
        else:
            print(self.imglist[index],11)
        # mask
        mask = self.generate_stroke_mask([self.image_szie, self.image_szie])
        mask[:,:,0] = mask[:,:,0]*(img[:,:]>0)

        while ((mask[:,:,0] == 1).sum() / (img[:,:]!=0).sum()) < 0.2 or ((mask[:,:,0] == 1).sum() / (img[:,:]!=0).sum()) > 0.6:
            mask = self.generate_stroke_mask([self.image_szie, self.image_szie])
            mask[:,:,0] = mask[:,:,0]*(img[:,:]>0)
        # the outputs are entire image and mask, respectively
        # img = torch.from_numpy(img.astype(np.float32) / 255.0).contiguous()
        # mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])
        compose1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
        img = compose(img).float()
        # img = (img-img.min()) / (img.max() - img.min())

        return img, compose(mask)

    def generate_stroke_mask(self, im_size, parts=7, maxVertex=20, maxLength=80, maxBrushWidth=80, maxAngle=360):
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask

    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int_)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int_)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        return mask
    
class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self,path,mask_path,imgsize = 256):
        self.path = path
        self.namelist = []
        self.mask_path = mask_path
        self.imgsize = imgsize
        for filespath in os.listdir(path):
            self.namelist.append(filespath)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        imgpath = os.path.join(self.path, imgname)
        img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.imgsize, self.imgsize))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # mask
        maskpath = os.path.join(self.mask_path, imgname)
        maskpath = maskpath.replace('t2','seg')
        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.imgsize, self.imgsize))
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return img, mask,