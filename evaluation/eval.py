import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sep
from photutils.profiles import CurveOfGrowth
# from photutils.centroids import centroid_2dg
from skimage.feature import peak_local_max
from photutils.morphology import data_properties
from astropy.table import vstack


def _FindCenterPeak(image):
    cenxy = peak_local_max(image,threshold_abs=0.2,num_peaks=1,exclude_border=10)
    return cenxy

def GetCenterPeak(images):
    n, h, w = images.shape
    cenxy = np.zeros((n,2))
    for ii in range(n):
        cenxy[ii] = _FindCenterPeak(images[ii])
    return cenxy

def normalize_unit(image):
    """Normalize the input image to be between 0 and 1."""
    im_norm = abs(image.min(axis=(1,2), keepdims=True)) + image
    im_norm = im_norm / im_norm.max(axis=(1,2), keepdims=True)
    return im_norm

def normalize_center(image, center = 35):
    """
    Normalize the input image to be between 0 and 1, using the center of the image as the reference point.
    The center of the image is defined as a square region of size `center x center`. Values above 1 are limited to 1. This function is useful for images where the center region is of interest and should be used as the reference for normalization.
    """
    n,h,w = image.shape
    imagecen = np.zeros((n,center,center))
    for i in range(n):
        imagecen[i] = image[i][math.ceil(h/2)-math.ceil(center/2):math.ceil(h/2)+math.ceil(center/2),math.ceil(w/2)-math.ceil(center/2):math.ceil(w/2)+math.ceil(center/2)]
    im_norm = abs(imagecen.min(axis=(1,2), keepdims=True)) + image
    im_norm = im_norm / imagecen.max(axis=(1,2), keepdims=True)
    return np.where(im_norm<1,im_norm,1) # limit any values above 1 to be set at the maximum which is 1

def ZScoreNormalize(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each channel of the input image independently.
    
    Parameters:
    image (np.ndarray): Input image of shape (C, H, W) where C is the number of channels.
    
    Returns:
    np.ndarray: Z-score normalized image of the same shape as the input.
    """
    mean = np.mean(image, axis=(1, 2), keepdims=True)
    std = np.std(image, axis=(1, 2), keepdims=True)
    return (image - mean) / std

# taken from BTK reconstruction eval
def psnr(images1: np.ndarray, images2: np.ndarray) -> np.ndarray:
    """Compute peak-signal-to-noise-ratio from skimage.

    Args:
        images1: Array of shape `NHW` containing `N` images each of
                size `NHW`.
        images2: Array of shape `NHW` containing `N` images each of
                size `NHW`.

    Returns:
        Returns PSNR between each corresponding iamge as an array of shape `N`.
    """
    n, h, w = images1.shape
    assert images1.min() >= 0 and images2.min() >= 0
    assert (n, h, w) == images2.shape
    psnr_ = np.zeros(n)
    for ii in range(n):
        im1 = images1[ii] / images1[ii].max()
        im2 = images2[ii] / images2[ii].max()
        psnr_[ii] = peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr_


def struct_sim(images1: np.ndarray, images2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute structural similarity index from skimage.

    By default, we normalize the images to be between 0 and 1. So that the
    `data_range` is 1.

    Args:
        images1: Array of shape `NHW` containing `N` images each of
                        size `NHW`.
        images2: Array of shape `NHW` containing `N` images each of
                        size `NHW`.
        kwargs: Keyword arguments to be passed in to `peak_signal_noise_ratio` function
                within `skimage.metrics`.

    Returns:
        Returns structural similarity index between each corresponding iamge as
        an array of shape `N`.
    """
    assert images1.min() >= 0 and images2.min() >= 0
    n, h, w = images1.shape
    assert (n, h, w) == images2.shape
    ssim = np.zeros(n)
    if 'full' in kwargs:
        Sims = np.zeros_like(images1)
    for ii in range(n):
        im1 = images1[ii] / images1[ii].max()
        im2 = images2[ii] / images2[ii].max()
        if 'full' in kwargs:
            ssim, S= structural_similarity(im1, im2, data_range=1, **kwargs)
            ssim[ii] = ssim
            Sims[ii] = S
        else:
            ssim[ii] = structural_similarity(im1, im2, data_range=1, **kwargs)
    if 'full' in kwargs:
        return ssim, Sims
    else:
        return ssim
    
# TODO: add aperture photometry metric
def _get_single_aperture_flux(image, x, y, radius, sky_level=None):
    """Compute the flux within a circular aperture centered at (x, y) with the given radius."""
    assert image.ndim == 2
    # assert x.ndim == 1 and y.ndim == 1
    flux, fluxerr, _ = sep.sum_circle(image, [x], [y], radius, var=sky_level)
    return flux, fluxerr

def get_aperture_fluxes(images, x, y, radius, sky_level=None):
    """Compute the flux within a circular aperture centered at (x, y) with the given radius for each image."""
    n, h, w = images.shape
    assert x.shape == (n,) and y.shape == (n,)
    fluxes = np.zeros(n)
    fluxerrs = np.zeros(n)
    for ii in range(n):
        fluxes[ii], fluxerrs[ii] = _get_single_aperture_flux(images[ii], x[ii], y[ii], radius, sky_level)
    return fluxes, fluxerrs

# TODO: add half-light radius metric
def _compute_halfLightR(image, xycen, radii=np.linspace(1,25,50), error=None):
    """Compute the half-light radius in pixels of a single image given the center and a set of radii."""
    
    cog = CurveOfGrowth(image,xycen,radii,error=error)
    cog.normalize(method='max')
    hfc = abs(cog.profile-0.5).argmin()
    return cog.radius[hfc]

def get_HLR(images, x, y):
    n, h, w = images.shape
    assert x.shape == (n,) and y.shape == (n,)
    hlrs = np.zeros(n)
    for ii in range(n):
        hlrs[ii] = _compute_halfLightR(images[ii], (x[ii], y[ii]))
    return hlrs

def _match_segmentation(image, catalog):
    cenxy = _FindCenterPeak(image)
    cenx,ceny = cenxy[0][0],cenxy[0][1]
    mdist = 0
    inds = 0
    for n in range(len(catalog)):
        cat_x,cat_y = catalog[n]['x'], catalog[n]['y']
        dist = np.sqrt((cat_x - cenx)**2 + (cat_y - ceny)**2)
        if n == 0:
            mdist = dist
            inds = 0
        else:
            if dist < mdist:
                mdist = dist
                inds = n
    return inds


def _compute_segmentation(image,thresh,minarea):
    """Perform segmentation on a single image."""
    bkg = sep.Background(image)
    catalog, segmentation = sep.extract(
            image,
            thresh=thresh,
            err=bkg.globalrms,
            segmentation_map=True,
            minarea=minarea,
        )
    inds = _match_segmentation(image,catalog)
    seg_i = segmentation == inds + 1
    return seg_i

def get_segmentation(images, thresh = 5, minarea=5):
    """Perform segmentation on each image in the batch."""
    n, h, w = images.shape
    segments = np.zeros((n, h, w), dtype=bool)
    for ii in range(n):
        segments[ii] = _compute_segmentation(images[ii],thresh,minarea)
    return segments

# taken from BTK segmentation eval
def iou(seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
    """Calculates intersection-over-union (IoU) given two semgentation arrays.

    This metric assumes that the input arrays are boolean. Otherwise the arrays are
    casted to boolean arrays before the computation.

    Args:
        seg1: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.
        seg2: Array of shape `NHW` containing `N` segmentation maps each of
                        size `HW`.

    Returns:
        Returns `iou` between each corresponding segmentation map as an array of shape `N`

    """
    assert not np.any(np.isnan(seg1)) and not np.any(np.isnan(seg2))
    seg1 = seg1.astype(bool)
    seg2 = seg2.astype(bool)
    i = np.logical_and(seg1, seg2).sum(axis=(-1, -2))
    u = np.logical_or(seg1, seg2).sum(axis=(-1, -2))
    return i / u

def get_morph(images, use_seg=True):
    cols = ['semimajor_axis','semiminor_axis','orientation','ellipticity','kron_flux']
    # t1 = QTable(names= cols)
    n, h, w = images.shape
    t1 = None
    for i in range(n):
        if use_seg:
            segs = get_segmentation(np.array([images[i]]))
            dp = data_properties(images[i],mask=~segs[0])
        else:
            dp = data_properties(images[i])
        tbl = dp.to_table(columns=cols)
        if t1 is not None:
            t1 = vstack([t1,tbl],join_type='exact')
        else:
            t1 = tbl
    return t1
