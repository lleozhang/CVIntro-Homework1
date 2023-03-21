import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.

    x_grad = Sobel_filter_x(input_img)
    y_grad = Sobel_filter_y(input_img)
    
    I_xx, I_yy, I_xy = x_grad * x_grad, y_grad * y_grad, x_grad * y_grad
    window = np.ones((window_size, window_size))
    
    H, W = I_xx.shape
    KH, KW = window_size, window_size
    padded_size = (KH - 1) // 2
    
    
    I_xx = convolve(padding(I_xx, padded_size, "replicatePadding"), window)
    I_yy = convolve(padding(I_yy, padded_size, "replicatePadding"), window)
    I_xy = convolve(padding(I_xy, padded_size, "replicatePadding"), window)

    all_value = np.stack((I_xx, I_xy, I_xy, I_yy), axis = 2).reshape(H, W, 2, 2)
    det = np.linalg.det(all_value)
    tr = np.trace(all_value, axis1 = 2, axis2 = 3)
    
    theta = det - alpha * tr * tr
    
    ret = np.where(theta > threshold)
    theta_lis = theta[ret]
    corner_list = list(zip(ret[0], ret[1], theta_lis))
    
    return corner_list # the corners in corne_list: a tuple of (index of rows, index of cols, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
