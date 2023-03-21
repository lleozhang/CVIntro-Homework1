import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, padding
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    magnitude_grad = magnitude_grad / magnitude_grad.max() * 255
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    
    R = np.full_like(grad_mag, 255)
    Q = np.full_like(grad_mag, 255)
    deg_dir = grad_dir * 180. / np.pi
    np.where(deg_dir < 0, deg_dir + 180, deg_dir)
    padded_grad = padding(grad_mag, 1, "replicatePadding")
    
    R = np.where((deg_dir > 22.5) & (deg_dir <= 67.5), padded_grad[2 : , 2 : ], R)
    R = np.where((deg_dir > 67.5) & (deg_dir <= 112.5), padded_grad[2 : ,1 : -1], R)
    R = np.where((deg_dir > 112.5) & (deg_dir <= 157.5), padded_grad[: -2, 2 : ], R)
    R = np.where((deg_dir > 157.5) | (deg_dir <= 22.5), padded_grad[1 : -1, 2 : ], R)
    
    Q = np.where((deg_dir > 22.5) & (deg_dir <= 67.5), padded_grad[ : -2,  : -2], Q)
    Q = np.where((deg_dir > 67.5) & (deg_dir <= 112.5), padded_grad[ : -2 , 1 : -1], Q)
    Q = np.where((deg_dir > 112.5) & (deg_dir <= 157.5), padded_grad[2 : ,  : -2], Q)
    Q = np.where((deg_dir > 157.5) | (deg_dir <= 22.5), padded_grad[1 : -1,  : -2], Q)
    
    zeros = np.zeros_like(grad_mag)
    NMS_output = np.where(grad_mag > R, grad_mag, zeros)
    NMS_output = np.where(NMS_output > Q, NMS_output, zeros)
    
    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.01
    high_ratio = 0.2
    
    high_threshold = high_ratio * img.max()
    low_threshold = low_ratio * img.max()
    
    
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < low_threshold:
                output[i, j] = 0
            elif img[i, j] >= high_threshold:
                output[i, j] = 1
            else:
                for dx, dy in dirs:
                    tx = i + dx
                    ty = j + dy
                    if tx >= 0 and ty >=0 and tx < img.shape[0] and ty < img.shape[1]:
                        if img[tx, ty] >= high_threshold or output[tx, ty] == 1:
                            output[i, j] = 1
                            break
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
