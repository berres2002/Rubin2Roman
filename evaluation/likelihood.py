# import os
# import random as rd
# from typing import Callable, List, Optional, Tuple, Union

# import galsim
import numpy as np
# import surveycodex
# from astropy.io import fits
# from astropy.wcs import WCS
# from scipy import signal
# from scipy.interpolate import RegularGridInterpolator
# from skimage.measure import block_reduce
import pickle
import torch
import torch.nn as nn
# from torch.func import vjp

# img = torch.rand(1, 1, 64, 64)  # (batch, channels, H, W)



with open('/projects/bfpq/rubin2roman/psf_arr_y.pkl', 'rb') as f:
    # global gs_psf
    gs_psf = pickle.load(f)
    # gs_psf = gs_psf.withFlux(1.0)  # ensure unit flux
    # gs_psf = torch.from_numpy(gs_psf.drawImage(nx=64, ny=64, scale=0.11, method='auto').array)  # convert to torch tensor

mult_val = 1/159.23617710583153

class PhysModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.gs_psf = gs_psf
        self.mult_val = mult_val

    def convolve_with_psf(self,x, flip_kernel=True):
        """
        img: torch.Tensor, shape (N, C, H, W)
        psf_array: numpy array or torch.Tensor, shape (kh, kw) - single PSF applied per channel
        """
        # print("is_grad_enabled:", torch.is_grad_enabled())
        psf_array = self.gs_psf
        # device=self.device
        if isinstance(psf_array, np.ndarray):
            psf = torch.from_numpy(psf_array).float()
        else:
            psf = psf_array.float()
        psf = psf.to(self.device)

        # Normalize to unit flux (match GalSim's flux-normalized PSF convention)
        psf = psf / psf.sum()

        if flip_kernel:
            psf = torch.flip(psf, dims=[-2, -1])  # true convolution, not cross-correlation

        # N, C, H, W = img.shape
        N,C, H, W = x.shape
        kh, kw = psf.shape

        # Depthwise conv: apply same PSF independently to every channel
        weight = psf.expand(C, 1, kh, kw).contiguous()  # (C, 1, kh, kw)

        # Pad to preserve flux at edges — reflect avoids the zero-padding flux loss
        pad_h, pad_w = kh // 2, kw // 2
        img_padded = nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        x = nn.functional.conv2d(img_padded, weight, groups=C)
        return x

    def forward(self,x, ):
        # device=self.device
        # y = y#.to(device='cuda')
        # x=x.detach().cpu().numpy()
        # conv = nn.Conv2d(1, 1, kernel_size=64, stride=1, padding=0, bias=False
        # print(x.shape)
        # print("forward in: ", x.requires_grad, x.grad_fn)
        x = self.convolve_with_psf(x, flip_kernel=True)
        # print("forward out:", x.requires_grad, x.grad_fn)
        # galaxy = galsim.InterpolatedImage(galsim.Image(x, scale=0.11))
        # convolved = galsim.Convolve([galaxy, gs_psf])
        # result_image = convolved.drawImage(nx=64, ny=64, scale=0.2, method='auto')
        # result_array = result_image.array
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # result_array = torch.from_numpy(result_array)
        x = pool(x)
        # mu_t = torch.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
        # mu_t = torch.exp(0.25* t*(beta_min*(-2+t)-beta_max*t))
        # ln_score = y*mu_t - out
        # print("forward out:", x.requires_grad, x.grad_fn)
        return x

class likelihood:
    def __init__(self, rubin_im, rubin_var, batch_size, device='cuda'):
        rub_im=np.repeat(rubin_im[np.newaxis, :], batch_size, axis=0)
        yim=rub_im[:,-1][:,np.newaxis]
        self.y_in = torch.from_numpy(yim)
        y_var=np.repeat(rubin_var[np.newaxis, :,:], batch_size, axis=0)
        y_var=y_var[:][:,np.newaxis]
        self.y_var = torch.from_numpy(y_var)
        self.device = device

    def LL(self,x):
        y= self.y_in.to(device=self.device)
        var=self.y_var.to(device=self.device)
        A = PhysModel(device=self.device)
        Ax = A(x)
        return ((-0.5 * (y - Ax)**2)/ var)
    
    def score(self,x,t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            log_like = self.LL(x).sum()
            grad_x = torch.autograd.grad(log_like, x)[0]
        return grad_x