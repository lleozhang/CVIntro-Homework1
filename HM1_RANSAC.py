import numpy as np
from utils import draw_save_plane_with_points


if __name__ == "__main__":


    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    
    C = np.ones((noise_points.shape[0], 1))
    all_points = np.concatenate((noise_points, C), axis = -1)
    
    sample_time = int(np.ceil(np.log(0.001) / np.log(1 - (100 / 130) ** 3))) #more than 99.9% probability at least one hypothesis does not contain any outliers 
    distance_threshold = 0.05

    # sample points group

    sample_points = np.random.choice(noise_points.shape[0], sample_time * 3)
    samples = all_points[sample_points].reshape(sample_time, 3, 4)

    # estimate the plane with sampled points group

    _, __, svd = np.linalg.svd(samples)
    preds = svd[:, -1, :]


    #evaluate inliers (with point-to-plance distance < distance_threshold)

    div = np.sqrt((preds * preds).sum(axis = -1).reshape(-1, 1))
    dis = np.abs(preds @ all_points.T) / div
    
    count = np.zeros_like(dis)
    count[dis <= distance_threshold] = 1
    
    counts = count.sum(axis = -1)
    best_row = np.argmax(counts)
    inlier_index = np.where(count[best_row, :] == 1)

    inliers = all_points[inlier_index[0]]


    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 

    _, __, pred = np.linalg.svd(inliers)
    pf = pred[-1, :]


    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

