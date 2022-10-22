import numpy as np
from scipy import ndimage
import scipy.signal as sig
import nibabel as nib
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

path = 'C:/Users/Lenovo/Desktop/heart_data/'

def bbox(gt_1, gt_2, extra=10):
    
    # We have 2 segmentation instances for each patient. We want to use a single bbox for all the 10*30 images.
    # The segmentation pixels are either not 0 (where the heart is) or 0, we use this to get all the pixels where the heart should be.
    # From these pixels we can get the x,y boundaries of the box in which these pixels are
    # From these 4 numbers (y_min,y_max,x_min,x_max) we can construct the bbox
    
    if gt_1.shape != gt_2.shape:
        print('Ground truth image shape missmatch')
        
    max_list=[0,256,0,256]
    
    for y in range(gt_1.shape[0]):
        for x in range(gt_1.shape[1]):

            if gt_1[y][x][1] != 0 or gt_2[y][x][1] != 0:
                if y >= max_list[0]:
                    max_list[0] = y
                if y <= max_list[1]:
                    max_list[1] = y
                if x >= max_list[2]:
                    max_list[2] = x
                if x<= max_list[3]:
                    max_list[3] = x
                    
    max_list[0] = max_list[0] + extra
    max_list[1] = max_list[1] - extra
    max_list[2] = max_list[2] + extra
    max_list[3] = max_list[3] - extra
                    
    return max_list


def downsample2d(inputArray, kernelSize):
    
    #from https://gist.github.com/andrewgiessel/2955714?fbclid=IwAR2f3DG-PylDDxCEdby13Vzf93r_CLnMWCpO1_HOJqVLwZ7DhGixCeNExgE#file-gistfile1-py                    

    """This function downsamples a 2d numpy array by convolving with a flat
    kernel and then sub-sampling the resulting array.
    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.
    :param: inputArray: 2d numpy array
    :param: kernelSize: integer
    """
    average_kernel = np.ones((kernelSize,kernelSize))

    blurred_array = sig.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array

def downsample3d(inputArray, kernelSize):
    """This function downsamples a 3d numpy array (an image stack)
    by convolving each frame with a flat kernel and then sub-sampling the resulting array,
    re-building a smaller 3d numpy array.
    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.
    The array will be downsampled in the first 2 dimensions, as shown below.
    import numpy as np
    >>> A = np.random.random((100,100,20))
    >>> B = downsample3d(A, 2)
    >>> A.shape
    (100, 100, 20)
    >>> B.shape
    (50, 50, 20)
    :param: inputArray: 2d numpy array
    :param: kernelSize: integer
    """
    first_smaller = downsample2d(inputArray[:,:,0], kernelSize)
    smaller = np.zeros((first_smaller.shape[0], first_smaller.shape[1], inputArray.shape[2]))
    smaller[:,:,0] = first_smaller

    for i in range(1, inputArray.shape[2]):
        smaller[:,:,i] = downsample2d(inputArray[:,:,i], kernelSize)
    return smaller


def crop_img(img, bbox_boundaries, kernel_size=None):
    
    # crops the image based on the bbox
    
    if kernel_size == None:
        y_max = bbox_boundaries[0]
        y_min = bbox_boundaries[1]
        x_max = bbox_boundaries[2]
        x_min = bbox_boundaries[3]
        
    else:
        y_max = bbox_boundaries[0] / kernel_size
        y_min = bbox_boundaries[1] / kernel_size
        x_max = bbox_boundaries[2] / kernel_size
        x_min = bbox_boundaries[3] / kernel_size
        
    crop_img = np.zeros((img.shape[0],img.shape[1]))
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if y_min < y and y_max > y:
                if x_min < x and x_max > x:
                    crop_img[y][x] = img[y][x]
                    
    return crop_img

def data_transform_split(path, split, split_name):
    
    # we structure our data for each patient the following way:
    # each patient has a folder in which their 4d nifti file is split into 2d images, so if a patient has a 216x256x10x30 nifti file than the folder will contain 10x30 so 300 216x256 sized image
    # for training there is a text file in which each row contains the path for one image for one patient and the bbox coordinates for that image, the rows are separated by "/n"
    
    high_res_path_text_list = ""
    path_text_list = ""
    kernel_size = 2
    #for patient in os.listdir(path + 'training/'):
    for number in split:
            
        patient = "patient" + number
        patient_list = []
        nii_data = nib.load(os.path.join(path, 'training/', patient + '/', os.listdir(path + 'training/' + patient)[1])).get_fdata()

        Shape = list(nii_data.shape)
        Shape[0] = int(Shape[0] / (kernel_size))
        Shape[1] = int(Shape[1] / (kernel_size))

        downsample_data = np.zeros(Shape)

        for i in range(nii_data.shape[2]):
            for j in range(nii_data.shape[3]):

                downsample_data[:,:,i,j] = downsample2d(nii_data[:,:,i,j], kernel_size)

                seg_frame_1 = nib.load(os.path.join(path, 'training/', patient, os.listdir(path + 'training/' + patient)[3])).get_fdata()
                seg_frame_2 = nib.load(os.path.join(path, 'training/', patient, os.listdir(path + 'training/' + patient)[5])).get_fdata()

                bbox_boundaries = bbox(seg_frame_1, seg_frame_2)

                high_res_img = Image.fromarray(nii_data[:,:,i,j])
                high_res_img = high_res_img.convert("L")
                high_res_img.save(path + "training_2/" + patient + "/" + patient + str(i + 1) + "_" + str(j + 1) +".jpg")
                high_res_text = patient + "/" + patient + str(i + 1) + "_" + str(j + 1) +".jpg"

                bbox_text = ""
                bbox_text = str(bbox_boundaries[0]) + "," + str(bbox_boundaries[1]) + "," + str(bbox_boundaries[2]) + "," + str(bbox_boundaries[3])

                path_text_list += high_res_text + "," + bbox_text + "/n"
     

    with open(path + "training_2/"+ split_name + "_data.txt", "w") as file:
        file.write(path_text_list)
        
def patient_train_split(train_split=70, val_split=20, test_split=10):
    
    #70/20/10
    
    patient_list =  [i for i in range(100)]
    for i in range(len(patient_list)):
        if i < 10:
            patient_list[i] = '00' + str(i)
        if 10 <= i < 100:
            patient_list[i] = '0' + str(i)
        if i == 100:
            patient_list[i] = str(i)
            
    random.shuffle(patient_list)

    train_list = patient_list[:train_split]
    val_list = patient_list[val_split:]
    test_list = patient_list[train_split:test_split]
    
    return (train_list, val_list, test_list)
