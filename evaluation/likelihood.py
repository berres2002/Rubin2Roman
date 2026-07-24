import os
import random as rd
# from typing import Callable, List, Optional, Tuple, Union

import galsim
import numpy as np
import surveycodex
# from astropy.io import fits
# from astropy.wcs import WCS
# from scipy import signal
# from scipy.interpolate import RegularGridInterpolator
from skimage.measure import block_reduce
import pickle
import torch
import torch.nn as nn

img = torch.rand(1, 1, 64, 64)  # (batch, channels, H, W)



with open('gs_y_psf.pkl', 'rb') as f:
    global gs_psf
    gs_psf = pickle.load(f)
    gs_psf = gs_psf.withFlux(1.0)  # ensure unit flux
    gs_psf = torch.from_numpy(gs_psf.drawImage(nx=64, ny=64, scale=0.11, method='auto').array)  # convert to torch tensor

def convolve_with_psf(img, psf_array=gs_psf, flip_kernel=True):
    """
    img: torch.Tensor, shape (N, C, H, W)
    psf_array: numpy array or torch.Tensor, shape (kh, kw) - single PSF applied per channel
    """
    if isinstance(psf_array, np.ndarray):
        psf = torch.from_numpy(psf_array).float()
    else:
        psf = psf_array.float()

    # Normalize to unit flux (match GalSim's flux-normalized PSF convention)
    psf = psf / psf.sum()

    if flip_kernel:
        psf = torch.flip(psf, dims=[-2, -1])  # true convolution, not cross-correlation

    N, C, H, W = img.shape
    kh, kw = psf.shape

    # Depthwise conv: apply same PSF independently to every channel
    weight = psf.expand(C, 1, kh, kw).contiguous()  # (C, 1, kh, kw)

    # Pad to preserve flux at edges — reflect avoids the zero-padding flux loss
    pad_h, pad_w = kh // 2, kw // 2
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    out = F.conv2d(img_padded, weight, groups=C)
    return out

def ln_score_vp(t,x,y, beta_min, beta_max):
    # x=x.detach().cpu().numpy()
    # conv = nn.Conv2d(1, 1, kernel_size=64, stride=1, padding=0, bias=False)
    x = convolve_with_psf(x, gs_psf, flip_kernel=True)
    # galaxy = galsim.InterpolatedImage(galsim.Image(x, scale=0.11))
    # convolved = galsim.Convolve([galaxy, gs_psf])
    # result_image = convolved.drawImage(nx=64, ny=64, scale=0.2, method='auto')
    # result_array = result_image.array
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    # result_array = torch.from_numpy(result_array)
    out = pool(x)
    mu_t = np.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
    ln_score = y*mu_t - out
    
    return ln_score