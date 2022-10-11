def bbox(img,gt_1,gt_2,extra=10):
    
    if img.shape != gt_1.shape or img_shape != gt_2.shape:
        szoveg = "Panninak nem tetszik"
        print(szoveg)
        return None
    
    max_list=[0,256,0,256]
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if gt_1[y][x][1] != 0 or gt_2[y][x][1] != 0:
                if y >= max_list[0]:
                    max_list[0] = y
                if y <= max_list[1]:
                    max_list[1] = y
                if x >= max_list[2]:
                    max_list[2] = x
                if x<= max_list[3]:
                    max_list[3] = x
                    
    y_max = max_list[0] + extra
    y_min = max_list[1] - extra
    x_max = max_list[2] + extra
    x_min = max_list[3] - extra
    
    crop_img = np.zeros((img.shape[0],img.shape[1]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if y_min < y and y_max > y:
                if x_min < x and x_max > x:
                    crop_img[y][x] = img[y][x]
#from https://gist.github.com/andrewgiessel/2955714?fbclid=IwAR2f3DG-PylDDxCEdby13Vzf93r_CLnMWCpO1_HOJqVLwZ7DhGixCeNExgE#file-gistfile1-py                    
def downsample2d(inputArray, kernelSize):
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
