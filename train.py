from Generator import GatedGenerator
import torch
from torch.nn import init
import torchvision.models as models
from PerceptualNet import PerceptualNet
from Discriminator import PatchDiscriminator
from torchvision import transforms
import os
import time
import datetime
import cv2
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datasets import InpaintDataset
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

def gradient_penalty (netD, real_data, fake_data, mask):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD.forward(interpolates, mask)

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty

def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def create_generator(generator_model_path,resume = True):
    # Initialize the networks
    generator = GatedGenerator()
    print('Generator is created!')
    if resume == True:
        generator.load_state_dict(torch.load(generator_model_path))
        print('Load generator %s' % generator_model_path)
    else:
        # Init the networks
        weights_init(generator, init_type = 'kaiming', init_gain = 0.2)
        print('Initialize generator with %s type' % 'kaiming')
    return generator

def create_discriminator(discriminator_model_path,resume = True):
    # Initialize the networks
    discriminator = PatchDiscriminator()
    print('Discriminator is created!')
    if resume == True:
        discriminator.load_state_dict(torch.load(discriminator_model_path))
        print('Load generator %s' % discriminator_model_path)
    else:
        weights_init(discriminator, init_type = 'kaiming', init_gain = 0.2)
        print('Initialize discriminator with %s type' % 'kaiming')
    return discriminator

def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = PerceptualNet()
    # Pre-trained VGG-16
    try:
        vgg16 = torch.load('./vgg16_pretrained.pth')
    except:
        vgg16 = models.vgg16(pretrained=True)
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def trainer(data_path,generator_model_path,discriminator_model_path,lr_g=1e-4,lr_d=1e-4,batch_size=4,epochs=40,resume = True):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = True
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # configurations
    save_folder = 'models'
    sample_folder = 'samples'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = create_generator(generator_model_path,resume)
    discriminator = create_discriminator(discriminator_model_path,resume)
    perceptualnet = create_perceptualnet()

    # To device
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()#reduce=False, size_average=False)
    RELU = nn.ReLU()

    # Optimizers
    optimizer_g1 = torch.optim.Adam(generator.coarse.parameters(), lr=lr_g)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = lr_d)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch,batch=0, is_D=False):
        """Save the model at "checkpoint_interval" and its multiple"""
        if is_D==True:
            model_name = 'discriminator.pth'
        else:
            model_name = 'deepfill.pth'
        model_name = os.path.join(save_folder, model_name)
        if epoch % 1 == 0:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d batch %d' % (epoch, batch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = InpaintDataset(path=data_path,image_szie = 256)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(epochs):
        print("Start epoch ", epoch+1, "!")
        for batch_idx, (img, mask) in enumerate(dataloader):

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img = img.cuda()
            mask = mask.cuda()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            if (batch_idx+1) % 10000 != 0:
                optimizer_d.zero_grad()
                fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
                true_scalar = discriminator(img, mask)
                W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)+ gradient_penalty(discriminator, img, second_out_wholeimg, mask)
                hinge_loss = torch.mean(RELU(1-true_scalar)) + torch.mean(RELU(fake_scalar+1))
                loss_D = hinge_loss
                loss_D.backward(retain_graph=True)
                optimizer_d.step()


            ### Train Generator
            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = - torch.mean(fake_scalar)

            optimizer_g1.zero_grad()
            first_MaskL1Loss.backward(retain_graph=True)
            optimizer_g1.step()

            optimizer_g.zero_grad()

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(torch.concat([img,img,img],dim=1))  # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(torch.concat([second_out_wholeimg,second_out_wholeimg,second_out_wholeimg],dim=1))
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            loss = 1 * first_MaskL1Loss +  0.5 *  second_MaskL1Loss + GAN_Loss + 256 * second_PerceptualLoss
            # loss = L_rec + 1e-4*GAN_Loss
            loss.backward()

            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()


            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                  ((epoch + 1), epochs, (batch_idx+1), len(dataloader), first_MaskL1Loss.item(),
                   second_MaskL1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                  (loss_D.item(), GAN_Loss.item(), time_left))

            if (batch_idx + 1) % 50 ==0:

                # Generate Visualization image
                ori_img = img * (1 - mask) + mask 
                back_img = transforms.RandomHorizontalFlip(p=1)(ori_img)
                masked_img = ori_img * (1 - mask) + mask  
                # masked_img = img * (1 - mask) + mask
                img_save = torch.cat((img, masked_img, first_out, first_out_wholeimg,second_out, second_out_wholeimg),3)
                # Recover normalization: * 255 because last layer is sigmoid activated
                # img_save = F.interpolate(img_save, scale_factor=0.5)
                img_save = img_save * 255
                # Process img_copy and do not destroy the data of img
                img_copy = img_save.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
                #img_copy = np.clip(img_copy, 0, 255)
                img_copy = img_copy.astype(np.uint8)
                save_img_name = 'sample_batch' + str(batch_idx+1) + '.png'
                save_img_path = os.path.join(sample_folder, save_img_name)
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, img_copy)

        #Learning rate decrease
        adjust_learning_rate(lr_g, optimizer_g, (epoch + 1))
        adjust_learning_rate(lr_d, optimizer_d, (epoch + 1))

        # Save the model
        save_model(generator, epoch)
        save_model(discriminator, epoch , is_D=True)