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