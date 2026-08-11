import os
# import random as rd
# from typing import Callable, List, Optional, Tuple, Union

# import galsim
import numpy as np
# import surveycodex
# from astropy.io import fits
# from astropy.wcs import WCS
# from scipy import signal
# from scipy.interpolate import RegularGridInterpolator
from skimage.measure import block_reduce
import pickle
import torch
import torch.nn as nn
from torch.func import vjp

# img = torch.rand(1, 1, 64, 64)  # (batch, channels, H, W)



with open('/projects/bfpq/rubin2roman/psf_arr_y.pkl', 'rb') as f:
    # global gs_psf
    gs_psf = pickle.load(f)
    # gs_psf = gs_psf.withFlux(1.0)  # ensure unit flux
    # gs_psf = torch.from_numpy(gs_psf.drawImage(nx=64, ny=64, scale=0.11, method='auto').array)  # convert to torch tensor

mult_val = 1/159.23617710583153

class likelihood:
    def __init__(self, rubin_im, batch_size, device='cuda'):
        rub_im=np.repeat(rubin_im[np.newaxis, :], batch_size, axis=0)
        yim=rub_im[:,-1][:,np.newaxis]
        self.y_in = torch.from_numpy(yim)
        self.device = device
        self.gs_psf = gs_psf
        self.mult_val = mult_val

    def convolve_with_psf(self,img, flip_kernel=True):
        """
        img: torch.Tensor, shape (N, C, H, W)
        psf_array: numpy array or torch.Tensor, shape (kh, kw) - single PSF applied per channel
        """
        psf_array = self.gs_psf
        device=self.device
        if isinstance(psf_array, np.ndarray):
            psf = torch.from_numpy(psf_array).float()
        else:
            psf = psf_array.float()
        psf = psf.to(device=device)

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
        img_padded = nn.functional.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        out = nn.functional.conv2d(img_padded, weight, groups=C)
        return out

    def A(self,x, ):
        # device=self.device
        # y = y#.to(device='cuda')
        # x=x.detach().cpu().numpy()
        # conv = nn.Conv2d(1, 1, kernel_size=64, stride=1, padding=0, bias=False)
        x = self.convolve_with_psf(x, flip_kernel=True)
        # galaxy = galsim.InterpolatedImage(galsim.Image(x, scale=0.11))
        # convolved = galsim.Convolve([galaxy, gs_psf])
        # result_image = convolved.drawImage(nx=64, ny=64, scale=0.2, method='auto')
        # result_array = result_image.array
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # result_array = torch.from_numpy(result_array)
        out = pool(x)
        # mu_t = torch.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
        # mu_t = torch.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
        # ln_score = y*mu_t - out
        
        return out

    def score(self,t,x,beta_min=1e-2, beta_max=20):
        y = self.y_in.to(device=self.device)
        y_hat, vjpf = vjp(self.A,x)
        # print(y_hat.shape)
        mu_t = torch.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
        # print(mu_t.shape)
        ln_score = mu_t[0]*y - y_hat
        return vjpf(ln_score)[0]