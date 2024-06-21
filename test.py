import cv2
import numpy as np
import torch
from Generator import GatedGenerator
from Discriminator import PatchDiscriminator
import matplotlib.pyplot as plt
from scipy import ndimage
from torchvision import transforms

def inference_Pipeline(img,mask,generator_model_path,discriminator_model_path):
    # img and mask
    imgs = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype('float32')
    masks = cv2.imread(mask,cv2.IMREAD_GRAYSCALE).astype('float32')
    imgs = cv2.resize(imgs, (256, 256))
    masks = cv2.resize(masks, (256, 256))
    masks = expend(np.clip(masks, 0, 1))*np.clip(imgs, 0, 1)
    compose1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    img = compose1(imgs).float()
    img = (img-img.min()) / (img.max() - img.min())
    # masks = masks[:,:,np.newaxis]
    img_tensor = img.unsqueeze(0)
    mask_tensor = torch.from_numpy(masks.astype(np.float32)).contiguous().unsqueeze(0).unsqueeze(0)
    # model
    generator = GatedGenerator()
    generator.load_state_dict(torch.load(generator_model_path))
    discriminator = PatchDiscriminator()
    discriminator.load_state_dict(torch.load(discriminator_model_path))
    
    generator.eval()
    discriminator.eval()
    # Generator output
    first_out, second_out = generator(img_tensor, mask_tensor)

    # forward propagation
    first_out_wholeimg = img_tensor * (1 - mask_tensor) + first_out * mask_tensor        # in range [0, 1]
    second_out_wholeimg = img_tensor * (1 - mask_tensor) + second_out * mask_tensor      # in range [0, 1]

    return second_out_wholeimg

def show(arry):
    plt.imshow(arry,cmap='gray')
    plt.show()

def expend(mmask):
    return ndimage.binary_dilation(mmask,iterations = 5).astype('uint16')
    