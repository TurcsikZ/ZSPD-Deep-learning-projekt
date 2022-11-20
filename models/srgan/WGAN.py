import os
import random
import time
import math
import numpy as np
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
#import skimage.measure as measure


class WGAN_GP(object): 
    """
    This class is for the mDCSRN+SRGAN network.
    Args:
    netG (model, or model.module) - Generator.
    netD (model, or model.module) - Discriminator.
    supervised_criterion (torch.nn.modules.loss) - the predefined loss function for the generator, in this project, we use nn.L1Loss().
    D_criterion (torch.nn.modules.loss) - the predefined loss function for the pretraining of discriminator. The idea is to let the discriminator first become a good classifier. So, we use nn.BCELoss().
    device (torch.device) - the device you set.
    ngpu (int) - how many GPU you use.
    lr (float) - the learning rate for pretraining. By default, the value is 5e-6.
    joint_opt_param (float) - the \lambda in the loss function. By default, the value is 0.001.
   """
    def __init__(self, netG, netD, 
             supervised_criterion, D_criterion, 
             device,gpu=False,
             lr=5e-6, joint_opt_param=0.001):
        self.netG = netG
        self.netD = netD
        self.supervised_criterion = supervised_criterion
        self.D_criterion = D_criterion
        self.device = "cuda:0"
        self.lr = lr
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr)
        self.optimizer_preD = optim.Adam(self.netD.parameters(),lr=self.lr)
        self.lmda = joint_opt_param   
        self.gpu = gpu
    
    def wasserstein_loss(self, D_fake, D_real= torch.Tensor([0.0]),gpu=False):
        if self.gpu:

            device = "cuda:0"

            D_real = D_real.cuda(self.device)
            D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
            G_loss = - torch.mean(D_fake)
            
        else:
            D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
            G_loss = - torch.mean(D_fake)

        return G_loss, D_loss
    
    def pre_updateD(self, lr_patches, hr_patches):
        '''
        This function completes the update of D network in D's pretraining. Note that we fix G and only train D, variables G_loss doesn't make sense in this function, so we set it to be 0.0. The value for loss you see is just the L1-loss.
        
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) D_loss: the Discriminator's loss.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss 
        '''
        # forward
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = True
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches.to(torch.float32))
        logit_fake = F.sigmoid(self.netD(sr_patches))
        # input HR to D (real)
        logit_real = F.sigmoid(self.netD(hr_patches))
        # Lable smoothing
#         fake = torch.tensor(torch.rand(logit_fake.size())*0.15)
#         real = torch.tensor(torch.rand(logit_real.size())*0.25 + 0.85)
        # Lable without smoothing
        fake = torch.tensor(torch.zeros(logit_fake.size()))
        real = torch.tensor(torch.ones(logit_real.size()))                
        if self.gpu:
            fake = fake.cuda(self.device)
            real = real.cuda(self.device)
        # discriminator loss for fake
        errD_fake = self.D_criterion(logit_fake, fake)
        errD_fake.backward()
        errD_real = self.D_criterion(logit_real, real)
        errD_real.backward()
        D_loss = errD_fake + errD_real
        self.optimizer_preD.step()
        G_loss = self.supervised_criterion(fake, fake) # here we don't count G, for G is not trained in WGAN yet
        # Supervised Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + self.lmda * G_loss
        
        return sr_patches, D_loss, G_loss, loss
    
    def updateD(self, lr_patches, hr_patches):
            '''
            This function completes the update of D network.

            (Input) lr_patches: the LR patches.
            (Input) hr_patches: the HR patches.
            (Output) sr_patches: the SR patches.
            (Output) D_loss: the Discriminator's loss.
            (Output) G_loss: the Generator's loss only for WGAN's part.
            (Output) loss: the network's loss 
            '''
            # forward
            for p in self.netG.parameters():
                p.requires_grad = True
            for p in self.netD.parameters():
                p.requires_grad = True
            # input SR to D (fake)
            sr_patches = self.netG(lr_patches)
            D_fake = self.netD(sr_patches)
            # input HR to D (real)
            D_real = self.netD(hr_patches)
            # Supervised Loss
            # Calculate L1 Loss
            L1_loss = self.supervised_criterion(sr_patches, hr_patches)
            # WGAN's Loss
            # Calculate Wasserstein Loss
            G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)
            # Semi-supervised Loss (main loss)
            loss = L1_loss + self.lmda * G_loss
            # backward + optimize only if in training phase
            D_loss.backward()
            self.optimizerD.step()

            # weight clipping
            for p in self.netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            return sr_patches, D_loss, G_loss, loss
        
    def updateG(self, lr_patches, hr_patches):
        '''
        This function completes the update of G network.
        
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss 
        '''
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD.parameters():
            p.requires_grad = False # to avoid computation
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss,_ = self.wasserstein_loss(D_fake)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + self.lmda * G_loss
        # backward + optimize only if in training phase
        loss.backward()
        self.optimizerG.step()
        return sr_patches, G_loss, loss
    
    
    def forwardDG(self, lr_patches, hr_patches):
        '''
        This function only goes through the forward of the network. It's used in validation period.
        
        (Input) lr_patches: the LR patches.
        (Input) hr_patches: the HR patches.
        (Output) sr_patches: the SR patches.
        (Output) D_loss: the Discriminator's loss.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) loss: the network's loss 
        '''
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # input HR to D (real)
        D_real = self.netD(hr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + G_loss
        return sr_patches, D_loss, G_loss, loss
    
    
    
    def training(self, dataloaders,
                 save_path="D:/WGAN_models/", max_step=550000, first_steps=10000, num_steps_pre = 250000,
                 usage=1.0, pretrainedG = ' ',pretrainedD =' '):
        #patch_size=1, cube_size=256,
        force_cudnn_initialization()
        since = time.time()
        print ("WGAN training...")
        print(self.gpu)
        if self.gpu:
            self.netG.cuda(self.device)
            self.netD.cuda(self.device)
        # record loss function of the whole period
        step = 0
        train_loss=[]
        train_D_loss=[]
        val_loss=[]
        val_D_loss=[]
        
        imbl = 1 # count for imbalance training
        extra = 1 # count for extra D training
        
        while(step < max_step):
            print('Step {}/{}'.format(step, max_step))
            print('-' * 10)
            mean_generator_content_loss = 0.0
            mean_discriminator_loss = 0.0
            # Each epoch has 10 training and validation phases
            for fold in range(10):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.netD.train()  # Set model to training mode
                        self.netG.train()
                    else:
                        self.netD.eval()   # Set model to training mode
                        self.netG.eval()
                    
                    batch_loss = []
                    batch_G_loss = []
                    batch_D_loss = []
                    val_ssim = []
                    val_psnr = []
                    val_nrmse = []
                    
                    for lr_data, hr_data in dataloaders[phase][fold]:
                        # This time, validation period would be different 
                        # since they need to be merged again to measure the evaluation metrics.
                        
                        #if phase == 'train':
                        #    patch_loader= dataloaders[phase][fold]
                        #else:
                        #    patch_loader=dataloaders[phase][fold]
                        #    sr_data_cat = torch.Tensor([]) # for concatenation
                            
                            
                        #for lr_patches, hr_patches in patch_loader:
                            
                        if self.gpu:
                            #print("ez jó")
                            lr_patches=lr_data.cuda(self.device)
                            hr_patches=hr_data.cuda(self.device)
                        else:
                            #print("ez jó")
                            lr_patches=lr_data
                            hr_patches=hr_data
                        # zero the parameter gradients
                        self.optimizerG.zero_grad()
                        self.optimizerD.zero_grad()

                        if phase == 'train':
                            # Training phase
                            with torch.set_grad_enabled(True):
                            ##########################################################
                            # (1) Update D network in following conditions:
                            #1.in first steps;
                            #2.every 500 steps for extra 200 steps;
                            #3.consecutive 7 steps.
                            # (2) Update G network in following conditions:
                            #1.after consecutive 7 steps Update D, update G for 1 step.
                            ##########################################################
                                # Update D Case 1: in first steps
                                if (step < num_steps_pre + first_steps):
                                    sr_patches, D_loss, G_loss, loss = self.pre_updateD(lr_patches, hr_patches)
                                    step += 1          # we count step here

                                # Regular training
                                else: 
                                    if ((imbl != 7) and (extra == 0)):
                                        # Update D Case 3: consecutive 7 steps
                                        sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                        step += 1
                                        imbl += 1
                                    if ((imbl == 7) and (extra == 0)):
                                        # Update G Case 1: update G for 1 step
                                        sr_patches, G_loss, loss = self.updateG(lr_patches, hr_patches)
                                        step += 1
                                        imbl = 1 # set to zero
                                    # Update D Case 2: every 500 steps for extra 200 steps
                                    if ((step % 500 == 0) or (extra != 0)):
                                        sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                        step += 1
                                        extra += 1
                                        if (extra == 200):
                                            extra = 1

                            #This print out is only for early inspection
                            if (step % 500) == 0:
                                print('Step: {}, loss= {:.4f}, D_loss= {:.4f}, G_loss= {:.4f}'.format(step, loss.item(), D_loss.item(), G_loss.item()))
                                step_time = time.time() - since
                                print("Time elapsed: {:.0f} seconds".format(step_time))

                            # statistics
                            batch_loss = np.append(batch_loss, loss.item())
                            batch_G_loss = np.append(batch_G_loss, G_loss.item())
                            batch_D_loss = np.append(batch_D_loss, D_loss.item())

                            #if ((step - num_steps_pre) % int((max_step - num_steps_pre) // 10)) ==0:
                            if step % 500 == 0:
                                # save intermediate models for singal GPU and multi GPU

                                torch.save(self.netG.state_dict(),save_path + 'pre_WGAN_G_step{}'.format(step))
                                torch.save(self.netD.state_dict(),save_path + 'pre_WGAN_D_step{}'.format(step))

                                # record instant loss
                                train_loss = np.append(train_loss, batch_loss)
                                train_D_loss = np.append(train_D_loss, batch_D_loss)

                            if (step == max_step):
                                print("True")
                                # record instant loss
                                train_loss = np.append(train_loss, batch_loss)
                                train_D_loss = np.append(train_D_loss, batch_D_loss)    

                                torch.save(self.netG.state_dict(),save_path + 'pre_final_model_G')
                                torch.save(self.netD.state_dict(),save_path + 'pre_final_model_D')

                                return self.netG, self.netD

                        else:
                            # Validation phase
                            with torch.set_grad_enabled(False):
                                sr_patches, D_loss, G_loss, loss = self.forwardDG(lr_patches, hr_patches)
                            # statistics
                            batch_loss = np.append(batch_loss, loss.item())
                            batch_G_loss = np.append(batch_G_loss, G_loss.item())
                            batch_D_loss = np.append(batch_D_loss, D_loss.item())
                            # concatenate patches, send patches to cpu to save GPU memory
                            #sr_data_cat = torch.cat([sr_data_cat, sr_patches.to("cpu")],0)
                                                        
                        if phase == 'val':
                            None
                            # calculate the evaluation metric
                            #sr_data = depatching(sr_data_cat, lr_data.size(0))
                            #f=open('example_images/image_lr_step{}.txt'.format(step),'wb')
                            #pickle.dump(lr_data.cpu().numpy() ,f)
                            #f.close()
                            #f=open('example_images/image_sr_step{}.txt'.format(step),'wb')
                            #pickle.dump(sr_data.cpu().numpy() ,f)
                            #f.close()
                            #f=open('example_images/image_hr_step{}.txt'.format(step),'wb')
                            #pickle.dump(hr_data.cpu().numpy() ,f)
                            #f.close()
                            #batch_ssim = ssim(hr_data, sr_data)
                            #batch_psnr = psnr(hr_data, sr_data)
                            #batch_nrmse = nrmse(hr_data, sr_data)
                            #val_ssim = np.append(val_ssim, batch_ssim)
                            #val_psnr = np.append(val_psnr, batch_psnr)
                            #val_nrmse = np.append(val_nrmse, batch_nrmse)
                    
                    mean_generator_content_loss = np.mean(batch_loss)
                    mean_discriminator_loss = np.mean(batch_D_loss)
                    if phase == 'val':
                        #mean_ssim = np.mean(val_ssim)
                        #std_ssim = np.std(val_ssim)
                        #mean_psnr = np.mean(val_psnr)
                        #std_psnr = np.std(val_psnr)
                        #mean_nrmse = np.mean(val_nrmse)
                        #std_nrmse = np.std(val_nrmse)
                        val_loss = np.append(val_loss, batch_loss)
                        val_D_loss = np.append(val_D_loss, batch_D_loss)
                        
                        #print('No. {} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}.'.format(fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                        #print('Metrics: subject-wise mean SSIM = {:.4f}, std = {:.4f}; mean PSNR = {:.4f}, std = {:.4f}; mean NRMSE = {:.4f}, std = {:.4f}.'.format(mean_ssim, std_ssim, mean_psnr, std_psnr, mean_nrmse, std_nrmse))
                    else:
                        train_loss = np.append(train_loss, batch_loss)
                        train_D_loss = np.append(train_D_loss, batch_D_loss)
                        print('No.{} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}'.format(fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                        
                        
                time_elapsed = time.time() - since
                print('Now the training uses {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
                print()
        return self.netG, self.netD

def ssim(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_ssim function to compute the mean structural similarity index between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) ssim: an ndarray with length (B,1), which contains the ssim value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    
    ssim=[]
    #for i in range(img_true.shape[0]):
        #ssim = np.append(ssim, measure.compare_ssim(img_true[i], img_test[i]))
    return ssim

def psnr(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_psnr function to compute the peak signal to noise ratio (PSNR) between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) psnr: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    psnr=[]
    #for i in range(img_true.shape[0]):
        #psnr = np.append(psnr, measure.compare_psnr(img_true[i], img_test[i]))
    return psnr

def nrmse(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_nrmse function to compute the normalized root mean-squared error (NRMSE) between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) nrmse: an ndarray with length (B,1), which contains the psnr value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    nrmse=[]
    #for i in range(img_true.shape[0]):
        #nrmse = np.append(nrmse, measure.compare_nrmse(img_true[i], img_test[i]))
    return nrmse 
    
    
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda:0')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))