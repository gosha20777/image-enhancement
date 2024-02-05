import math

import numpy as np
from scipy import stats as st
from scipy import signal
from scipy.ndimage.filters import convolve
import torch
from torch.nn import functional as F

from .config import config


def get_content(vgg19, img_tensor, content_id, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    content = vgg19.features[:content_id]((img_tensor - mean) / std)

    return content


def get_gaussian_kernel(kernel_size, sigma, channels, device):
    interval = (2 * sigma + 1) / kernel_size
    x = np.linspace(-sigma - interval / 2, sigma + interval / 2, kernel_size + 1)

    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    out_filter = np.float32(kernel)
    out_filter = out_filter.reshape((1, 1, kernel_size, kernel_size))
    out_filter = out_filter.repeat(channels, axis=0)
    out_filter = torch.as_tensor(out_filter, device=device)

    return out_filter


def gaussian_blur(img_tensor, kernel_size, sigma, channels, device):
    out_filter = get_gaussian_kernel(kernel_size, sigma, channels, device)

    return F.conv2d(img_tensor, out_filter, padding=kernel_size // 2, groups=channels)


def rgb_to_gray(img_tensor, device):
    rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)
    gray_image = torch.sum(img_tensor * rgb_weights, 1, keepdim=True)

    return gray_image


def psnr(image1, image2):
    """
    psnr : approximate estimate of absolute error
    image1 : (batch_size, channels, height, width) dslr image
    image2 : (batch_size, channels, height, width) enhanced image
    return : psnr_score of image1 and image2
    """
    image_size = config.channels * config.height * config.width
    #image1 = image1.view(config.batch_size, image_size)
    #image2 = image2.view(config.batch_size, image_size)
    # compute MSE with image1 and image2
    MSE = torch.sum(torch.pow((image1 - image2), 2)) / (config.batch_size * image_size)
    # compute psnr score
    psnr_score = 20 * math.log10(1) - 10 * math.log10(MSE)
    return psnr_score


def fspecial_gauss(window_size, window_sigma):
    """
    Function to mimic 'fspecial' of MATLAB function
    return : normalized gaussian kernel, mu_x and mu_y is set to 0
    """
    radius = window_size // 2
    offset = 0
    start, stop = -radius, radius + 1
    if window_size % 2 == 0:
        offset = 0.5
        stop = radius
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * window_sigma ** 2)))
    norm_kernel = kernel / kernel.sum()
    norm_kernel = torch.from_numpy(norm_kernel).float()
    return norm_kernel


def ssim(image1, image2, kernel_size=11, kernel_sigma=1.5):
    """
    ssim : consider image degradation as perceived change in structural information,
            while incorporating contrast masking and luminance masking
    image1 : (batch_size, channels, height, width) dslr image
    image2 : (batch_size, channels, height, width) enhanced image
    kernel_size : gaussian kernel size (window size of image)
    kernel_sigma : standard deviation of gaussian kernel
    """
    if type(image1) is not np.ndarray:
        image1 = image1.detach().cpu().numpy()
    if type(image2) is not np.ndarray:
        image2 = image2.cpu().numpy()

    # filter size can't be larger than height or width of images.
    filter_size = min(kernel_size, config.height, config.width)

    if filter_size:
        filter_sigma = filter_size * kernel_sigma / kernel_size
    else:
        filter_sigma = 0

    if kernel_size:
        window = np.reshape(fspecial_gauss(filter_size, filter_sigma), newshape=(1, 1, filter_size, filter_size))
        mu1 = signal.fftconvolve(image1, window, mode='same')
        mu2 = signal.fftconvolve(image2, window, mode='same')
        sigma11 = signal.fftconvolve(image1*image1, window, mode='same')
        sigma22 = signal.fftconvolve(image2*image2, window, mode='same')
        sigma12 = signal.fftconvolve(image1*image2, window, mode='same')
    else:  # empty gaussian blur kernel, no need to convolve
        mu1 = image1
        mu2 = image2
        sigma11 = image1 * image1
        sigma22 = image2 * image2
        sigma12 = image1 * image2

    mu_11 = mu1 * mu1
    mu_22 = mu2 * mu2
    mu_12 = mu1 * mu2
    sigma11 -= mu_11
    sigma22 -= mu_22
    sigma12 -= mu_12

    k_1, k_2 = 0.01, 0.03
    L = 255  # bitdepth of image, 2 ^ (bits per pixel) - 1
    c_1 = (k_1 * L) ** 2
    c_2 = (k_2 * L) ** 2

    v_1 = 2.0 * sigma12 + c_2
    v_2 = sigma11 + sigma22 + c_2

    ssim_score = np.mean(((2.0 * mu_12 + c_1) * v_1) / ((mu_11 + mu_22 + c_1) * v_2))
    cs_map = np.mean(v_1 / v_2)
    return ssim_score, cs_map


def multi_scale_ssim(image1, image2, kernel_size=11, kernel_sigma=1.5, weights=None):
    # default weights are None, but in below paper it has default weights
    # https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    if type(image1) is not np.ndarray:
        image1 = image1.detach().cpu().numpy()
    if type(image2) is not np.ndarray:
        image2 = image2.cpu().numpy()

    ms_ssim = np.array([])
    cs_map = np.array([])

    if weights:
        weights = np.array(weights)
    else:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    levels = len(weights)
    downsample = np.ones((1, 1, 2, 2)) / 4.0

    for i in range(levels):
        ssim_score, cs = ssim(image1, image2, kernel_size, kernel_sigma)
        ms_ssim = np.append(ms_ssim, ssim_score)
        cs_map = np.append(cs_map, cs)
        downsample_filtered = [convolve(image, downsample, mode='reflect') for image in [image1, image2]]
        image1, image2 = [image[:, :, ::2, ::2] for image in downsample_filtered]

    return np.prod(cs_map[0:levels-1] ** weights[0:levels-1]) * (ms_ssim[levels-1] ** weights[levels-1])
