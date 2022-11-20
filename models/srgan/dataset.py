import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, path, mode, pixelation=2):
        self.path = path
        if mode == "train":
            with open(self.path + "train.txt", "r") as f:
                self.text = f.read()
                self.img_path = path + "TRAIN/"
        if mode == "val":
            with open(self.path + "val.txt", "r") as f:
                self.text = f.read()
                self.img_path = path + "VAL/"
        if mode == "test":
            with open(self.path + "test.txt", "r") as f:
                self.text = f.read()
                self.img_path = path + "TEST/"
    
        self.data = self.text.split("\n")[:-1]
        self.pixelation = pixelation
        self.is_transform = True
        
    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).
        
        '''
        image_torch = torch.unsqueeze(torch.FloatTensor(image), 0)
        return image_torch
      
    def __getitem__(self, idx):
        img = Image.open(self.img_path + self.data[idx])
        #img = img.resize((256,256))
        image_lr = img
        image_hr = np.asarray(np.array(img))
        if self.pixelation==2:
            
            image_lr = two_pixelation2d(image_hr)
        if self.pixelation==3:
            
            image_lr = three_pixelation2d(image_hr)
            
        if(self.is_transform):
            sample_lr = self.transform(image_lr)
            sample_hr = self.transform(image_hr)
            
        return (sample_lr, sample_hr)
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, path, pixelation=2):
        self.data = [path]
        
        self.pixelation = pixelation
        self.is_transform = True
        
    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a torch.ShortTensor (B,z,x,y).
        
        '''
        image_torch = torch.unsqueeze(torch.FloatTensor(image), 0)
        return image_torch
      
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = img.resize((256,256))
        image_lr = img
        image_hr = np.asarray(np.array(img))
        if self.pixelation==2:
            
            image_lr = two_pixelation2d(image_hr)
        if self.pixelation==3:
            
            image_lr = three_pixelation2d(image_hr)
            
        if(self.is_transform):
            sample_lr = self.transform(image_lr)
            sample_hr = self.transform(image_hr)
            
        return (sample_lr, sample_hr)
    def __len__(self):
        return len(self.data)
def two_pixelation2d(img):
    i,j = img.shape
    x = np.zeros((i,j))
    n = 0
    while n <= j:
        for k in range(int(i/2)):

            #x[2*k][n] =
            K = int(2*k)
            try:
                x[K][n]  = img[K][n] 
                x[K+1][n]  = img[K][n] 
                x[K][n+1]  = img[K][n] 
                x[K+1][n+1] = img[K][n] 

            except IndexError:
                continue
        n += 2
    return x 

def three_pixelation2d(img):
    i,j = img.shape
    x = np.zeros((i,j))
    n = 0
    while n <= j:
        for k in range(int(i/3)):

            #x[2*k][n] =
            K = int(3*k)
            try:
                x[K][n]  = img[K][n] 
                x[K+1][n]  = img[K][n] 
                x[K][n+1]  = img[K][n] 
                x[K+1][n+1] = img[K][n]

            except IndexError:
                continue
        n += 3
    return x
    
def two_pixelation4d(img):
    i,j,t,f = img.shape
    x = np.zeros((i,j,t,f))
    n = 0
    for z in range(t):
        for l in range(f):
            while n <= j:
                for k in range(int(i/2)):

                    #x[2*k][n] =
                    K = int(2*k)
                    try:
                        x[K][n][z][l]  = img[K][n][z][l] 
                        x[K+1][n][z][l]  = img[K][n][z][l] 
                        x[K][n+1][z][l]  = img[K][n][z][l] 
                        x[K+1][n+1][z][l]  = img[K][n][z][l] 

                    except IndexError:
                        continue
                n += 2
    return x 

def three_pixelation4d(img):
    i,j,t,f = img.shape
    x = np.zeros((i,j,t,f))
    n = 0
    for z in range(t):
        for l in range(f):
            while n <= j:
                for k in range(int(i/3)):

                    #x[2*k][n] =
                    K = int(3*k)
                    try:
                        x[K][n][z][l] = img[K][n][z][l] 
                        x[K+1][n][z][l]  = img[K][n][z][l] 
                        x[K][n+1][z][l]  = img[K][n][z][l] 

                        x[K+1][n+1][z][l]  = img[K][n][z][l] 
                        x[K+2][n+1][z][l]  = img[K][n][z][l] 
                        x[K+1][n+2][z][l]  = img[K][n][z][l] 

                        x[K+2][n][z][l]  = img[K][n][z][l] 
                        x[K][n+2][z][l]  = img[K][n][z][l] 
                        x[K+2][n+2][z][l]  = img[K][n][z][l] 

                    except IndexError:
                        continue
                n += 3
    return x