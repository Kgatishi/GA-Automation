"""
This module is a collection of metrics to assess the similarity between two images.
Currently implemented metrics are FSIM, ISSM, PSNR, RMSE, SAM, SRE, SSIM, UIQ.
"""

import math

import numpy as np
from skimage.metrics import structural_similarity
import phasepack.phasecong as pc
import cv2



def rmse(org_img, pred_img, max_p= 4095):
    rmse_bands = []
    
    dif = np.subtract(org_img, pred_img)
    m = np.mean(np.square(dif / max_p))
    s = np.sqrt(m)
    rmse_bands.append(s)

    return np.mean(rmse_bands)


def psnr(org_img, pred_img, max_p= 255) :
    """
    Peek Signal to Noise Ratio, implemented as mean squared error converted to dB.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When using 12-bit imagery MaxP is 4095, for 8-bit imagery 255. For floating point imagery using values between
    0 and 1 (e.g. unscaled reflectance) the first logarithmic term can be dropped as it becomes 0
    """
    mse_bands = []
    
    mse_bands.append(np.mean(np.square(org_img - pred_img)))

    return 20 * np.log10(max_p) - 10.0 * np.log10(np.mean(mse_bands))

def ssim(org_img, pred_img, max_p= 255):
    return structural_similarity(org_img, pred_img, data_range=max_p)

def _similarity_measure(x, y, constant):
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant
    return numerator / denominator


def _gradient_magnitude(img, img_depth):
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)
    return np.sqrt(scharrx ** 2 + scharry ** 2)


def fsim(org_img, pred_img, T1 = 0.85, T2 = 160):


    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    
        # Calculate the PC for original and predicted images
    pc1_2dim = pc(org_img, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
    pc2_2dim = pc(pred_img, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
    pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
    pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
    for orientation in range(6):
        pc1_2dim_sum += pc1_2dim[4][orientation]
        pc2_2dim_sum += pc2_2dim[4][orientation]

    # Calculate GM for original and predicted images based on Scharr operator
    gm1 = _gradient_magnitude(org_img, cv2.CV_16U)
    gm2 = _gradient_magnitude(pred_img, cv2.CV_16U)

    # Calculate similarity measure for PC1 and PC2
    S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
    # Calculate similarity measure for GM1 and GM2
    S_g = _similarity_measure(gm1, gm2, T2)

    S_l = (S_pc ** alpha) * (S_g ** beta)

    numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    fsim_list.append(numerator / denominator)

    return (numerator / denominator)
    #return np.mean(fsim_list)