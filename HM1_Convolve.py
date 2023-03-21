import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":

        padding_img = np.zeros((img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size))
        padding_img[padding_size : -padding_size, padding_size : -padding_size] = img

        return padding_img
    
    elif type=="replicatePadding":

        padding_img = np.zeros((img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size))
        padding_img[padding_size : -padding_size, padding_size : -padding_size] = img
        
        padding_img[padding_size : -padding_size, 0 : padding_size] = img[:, 0].reshape(-1, 1)
        padding_img[padding_size : -padding_size, -padding_size : ] = img[:, -1].reshape(-1, 1)
        padding_img[0 : padding_size, padding_size : -padding_size] = img[0, :]
        padding_img[-padding_size : , padding_size : -padding_size] = img[-1, :]
        
        padding_img[0 : padding_size, 0 : padding_size] = img[0, 0]
        padding_img[-padding_size : , 0 : padding_size] = img[-1, 0]
        padding_img[0 : padding_size, -padding_size : ] = img[0, -1]
        padding_img[-padding_size : , -padding_size : ] = img[-1, -1]

        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, (img.shape[0] - kernel.shape[0] - 1) // 2, "zeroPadding")
    
    #build the Toeplitz matrix and compute convolution
    tgt_siz = padding_img.shape[0] - kernel.shape[0] + 1
    all_lines = np.arange(tgt_siz * tgt_siz).reshape(-1, 1)
    line_index = np.arange(tgt_siz * tgt_siz) // tgt_siz
    repeat_index = np.arange(tgt_siz).reshape(-1, 1).repeat(tgt_siz, axis = 1).T.reshape(-1, 1)

    ori_column_index = np.arange(kernel.shape[0])
    repeat_time = np.arange(kernel.shape[1])
    
    column_index = ori_column_index.reshape(-1, 1).repeat(kernel.shape[1], axis = 1).T + repeat_time.reshape(-1, 1)* padding_img.shape[1]
    
    all_column_index = column_index.reshape(-1, 1).repeat(tgt_siz * tgt_siz, axis = 1).T +  repeat_index + padding_img.shape[1] * line_index.reshape(-1, 1)

    TM = np.zeros((tgt_siz * tgt_siz,  padding_img.shape[0] * padding_img.shape[1]))
    TM[all_lines, all_column_index] = kernel.reshape(1, -1)
    
    output = (TM @ padding_img.reshape(-1, 1)).reshape(img.shape[0], img.shape[1])
    
    
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    fl_img = img.flatten()
    tgt_siz = img.shape[0] - kernel.shape[0] + 1
    
    ori_columns = np.arange(kernel.shape[1])
    repeat_columns = ori_columns.reshape(-1, 1).repeat(kernel.shape[0], axis = 1).T.reshape(1, -1)
    
    added = np.arange(tgt_siz).repeat(kernel.shape[0] * kernel.shape[1]).reshape(tgt_siz, -1)
    added_column = np.arange(kernel.shape[0]).repeat(kernel.shape[1])
    all_added = added.reshape(-1, 1).repeat(tgt_siz, axis = 1).T.reshape(tgt_siz * tgt_siz, -1)
    
    lines = np.arange(tgt_siz).repeat(tgt_siz).reshape(-1, 1)
    all_added = all_added + lines * img.shape[1]
    
    first_column = repeat_columns + added_column * img.shape[1]
    tmp_columns = first_column.reshape(-1, 1).repeat(tgt_siz * tgt_siz, axis = 1).T
        
    all_column = tmp_columns + all_added
    all_lines = np.arange(tgt_siz * tgt_siz)
    
    M = np.zeros((tgt_siz * tgt_siz, kernel.shape[0] * kernel.shape[1]))
    M[all_lines] = fl_img[all_column]
    
    output = (M @ kernel.reshape(-1, 1)).reshape(tgt_siz, tgt_siz)
    
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    