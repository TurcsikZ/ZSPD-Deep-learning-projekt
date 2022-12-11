import skimage.metrics as metrics
import torch
import numpy as np

def ssim(img_true, img_test):

    img_true = img_true.cpu().numpy()
    img_true = np.moveaxis(img_true, 1, 3)
    
    img_test = img_test.cpu().numpy()
    img_test = np.moveaxis(img_test, 1, 3)
    
    ssim=[]
    for i in range(img_true.shape[0]):
        ssim = np.append(ssim, metrics.structural_similarity(img_true[i], img_test[i],channel_axis=2))
    return ssim

def psnr(img_true, img_test,data_range=None):
    
    img_true = img_true.cpu().numpy().astype(float)/256
    img_true = np.moveaxis(img_true, 1, 3)
    
    img_test = img_test.cpu().numpy().astype(float)/256
    img_test = np.moveaxis(img_test, 1, 3)
                                                   
    psnr=[]
    for i in range(img_true.shape[0]):
        psnr = np.append(psnr, metrics.peak_signal_noise_ratio(img_true[i], img_test[i]))
    return psnr

def nrmse(img_true, img_test):
    
    img_true = img_true.cpu().numpy()
    img_true = np.moveaxis(img_true, 1, 3)
    
    img_test = img_test.cpu().numpy()
    img_test = np.moveaxis(img_test, 1, 3)
    
    nrmse=[]
    for i in range(img_true.shape[0]):
        nrmse = np.append(nrmse, metrics.normalized_root_mse(img_true[i], img_test[i]))
    return nrmse
