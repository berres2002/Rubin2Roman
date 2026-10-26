import torch
from torch import vmap
from astropy import units
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astropy.coordinates import SkyCoord
import torch.nn.functional as F
from definitions import interpolate, DEVICE


# with open('/projects/bfpq/rubin2roman/psf_arr_y.pkl', 'rb') as f:
#     # global gs_psf
#     gs_psf = pickle.load(f)
    # gs_psf = gs_psf.withFlux(1.0)  # ensure unit flux
    # gs_psf = torch.from_numpy(gs_psf.drawImage(nx=64, ny=64, scale=0.11, method='auto').array)  # convert to torch tensor

# 70x70 pixel PSF at 0.1 arcsec/pixel
with open('/projects/bfpq/rubin2roman/super_samp_rubin_psf_y70.npy', 'rb') as f:
    # global gs_psf
    gs_psf = np.load(f)

gs_psf = np.repeat(gs_psf[np.newaxis],3,axis=0)

def make_wcs(skycoord, orientation, pixels, pixel_size):
    """
    Create a World Coordinate System (WCS) object based on the given parameters.

    Parameters:
    skycoord (SkyCoord): The sky coordinates of the center of the image.
    orientation (float): The orientation of the image, defined as East of North in world coordinates.
    pixels (int): The number of pixels in each dimension of the image.
    pixel_size (Quantity): The size of each pixel in angular units.

    Returns:
    WCS: The World Coordinate System object.

    Raises:
    None.

    This function creates a FITS header and populates it with the necessary keywords to define a WCS.
    The FITS header is then used to create a WCS object, which can be used to convert between pixel coordinates and sky coordinates.

    The FITS header is populated with the following keywords:
    - NAXIS1: The number of pixels in the x-axis of the image.
    - NAXIS2: The number of pixels in the y-axis of the image.
    - CRVAL1: The right ascension of the center pixel of the image in degrees.
    - CRVAL2: The declination of the center pixel of the image in degrees.
    - CUNIT1: The units of the x-axis coordinates (degrees).
    - CUNIT2: The units of the y-axis coordinates (degrees).
    - CTYPE1: The coordinate type of the x-axis (RA---TAN).
    - CTYPE2: The coordinate type of the y-axis (DEC--TAN).
    - CDELTi: The pixel size of dimension 1 and 2 in degress
    - PCi_j: Elements of the pixel scale matrix for converting from pixel coordinates to sky coordinates.

    The PC matrix accounts for the mirror flip of the y-axis pixel coordinate (East<-West), thus 
    the PC matrix is a product of a pure rotation and a mirror flip of the second y-axis.

    The WCS object is created using the FITS header and returned.
    """
    hdr = fits.Header()
    hdr["NAXIS1"] = pixels
    hdr["NAXIS2"] = pixels
    hdr["CRVAL1"] = skycoord.ra.to(units.deg).value
    hdr["CRVAL2"] = skycoord.dec.to(units.deg).value
    hdr["CRPIX1"] = pixels/2 - 1 # Same convention as Cutout2D
    hdr["CRPIX2"] = pixels/2 - 1 
    hdr["CUNIT1"] = 'deg'
    hdr["CUNIT2"] = 'deg'
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CDELT1"] = 1.
    hdr["CDELT2"] = 1.
    cdelt = pixel_size.to(units.deg).value
    
    theta = orientation * np.pi / 180
    rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta), np.cos(theta)]])
    mirror_j = np.array([[1, 0], [0, -1]])
    pc = cdelt * rotation @ mirror_j
    hdr["PC1_1"] = pc[0, 0]
    hdr["PC1_2"] = pc[0, 1]
    hdr["PC2_1"] = pc[1, 0]
    hdr["PC2_2"] = pc[1, 1]
    return WCS(hdr)


def is_power_of_2(n):
    # if n is a power of 2, then n-1 flips all the bits in its binary rep. Thus, n AND n-1 will be 0 (all the bits will be different)
    if n <= 0:
        return False
    return n & (n - 1) == 0

def noise_padding(x, pad, sigma):
    B, C, H, W = x.shape
    PU, PD, PL, PR = pad
    out = torch.zeros(1, 1, H+PU+PD, W+PL+PR)
    # Put x in the center of the padded model
    out[..., PD:H+PU, PL:W+PR] = x
    # Create a mask for padding region
    mask = torch.ones_like(out)
    mask[..., PD:H+PU, PL:W+PR] = 0.
    # Noise pad around the model
    z = torch.randn_like(out) * sigma
    out += z * mask
    return out

def make_forward_model(
        psf:np.ndarray, 
        wcs_list:list[WCS, ...], 
        psf_super_sampling_factor:int,
        model_super_sampling_factor:int,
        model_pixels:int,
        model_pixel_size:units.Quantity,
        zero_padding:int=0,
        fiducial_center:SkyCoord=None,
        fiducial_orientation:float=None, # Pick the orientation of the first WCS, angle East of North
        sum_pool=False,
        **kwargs
        ):
    """
    Create a forward model for a given point spread function (PSF) and a list of world coordinate systems (WCS).

    Parameters:
    -----------
    psf : np.ndarray
        The point spread function (PSF) to be used for the forward model. It should be a 2D or 3D array (multi channel fit).

    wcs_list : list[WCS, ...]
        A list of world coordinate systems (astropy WCS) to be used for the forward model. 

    psf_super_sampling_factor : int
        The super sampling factor to be used for the forward model. It determines the level of detail in the model.

    model_super_sampling_factor : int
        The super sampling factor to be used for the forward model. It determines the level of detail in the model.

    model_pixels : int
        The number of pixels to be used for the model pixel grid.

    model_pixel_size : units.Quantity
        The size of each pixel in the model pixel grid. It should be an instance of the units.Quantity class.

    zero_padding : int, optional
        The number of zero padding pixels to be added to the model pixel grid on each side. Default is 0.

    fiducial_center : SkyCoord, optional
        The fiducial center to be used for the model reference pixel. It should be an instance of the SkyCoord class. 
        Default (None) is to use first WCS center pixel world coordinate.
        
    fiducial_orientation : float, optional
        The fiducial orientation to be used for model pixel grid. The angle is defined East of North. 
        Default (None) is to use first WCS pixel scale matrix orientation.

    Returns:
    --------
    forward_model : np.ndarray
        The created forward model as a 2D array.

    Examples:
    ---------
    >>> psf = np.ones((5, 5))
    >>> wcs_list = [WCS(), WCS()]
    >>> super_sampling_factor = 2
    >>> model_pixels = 10
    >>> model_pixel_size = units.Quantity(0.1, 'arcsec')
    >>> forward_model = make_forward_model(psf, wcs_list, super_sampling_factor, model_pixels, model_pixel_size)
    """

    if psf.ndim == 2:
        C = 1
        H, W = psf.shape
    elif psf.ndim == 3:
        C, H, W = psf.shape
    psf = torch.tensor(psf).float().to(DEVICE).view(C, 1, H, W) # reshape to a convolution kernel [channel_out, channels_in/groups, H, W]
    batched_interpolation = vmap(interpolate, in_dims=(0, None))  # only batch over the images (first argument of interpolate)
    
    # Create Fiducial WCS for the model
    if fiducial_center is None:
        # Use same convention as Cutout2D for reference pixel (assuming dim is even)
        center = [dim / 2 - 1  for dim in wcs_list[0].pixel_shape]
        fiducial_center = wcs_list[0].pixel_to_world(*center)
    if fiducial_orientation is None:
        pc = wcs_list[0].pixel_scale_matrix
        fiducial_orientation = np.arctan2(pc[1, 0], pc[0, 0]) * 180 / np.pi
    fiducial_wcs = make_wcs(fiducial_center, fiducial_orientation, model_pixels, model_pixel_size)
    print("Fiducial WCS")
    print(fiducial_wcs)
    

    model_kernel = model_super_sampling_factor / psf_super_sampling_factor
    ssf = psf_super_sampling_factor
    szp = ssf * zero_padding
    # Prepare Drizzle coordinate systems
    coordinates_list = []
    for wcs in wcs_list:
        u = np.arange(-szp, ssf * wcs.pixel_shape[0] + szp) / ssf
        v = np.arange(-szp, ssf * wcs.pixel_shape[1] + szp) / ssf
        u, v = np.meshgrid(u, v, indexing="ij")
        world = wcs.pixel_to_world(u, v)
        coordinates = np.stack(fiducial_wcs.world_to_pixel(world), axis=0)
        coordinates_list.append(torch.tensor(coordinates).float().to(DEVICE))
    
    def A(x):
        ys = []
        for i in range(len(wcs_list)):
            y = batched_interpolation(x, coordinates_list[i]) * (model_kernel**2 if sum_pool else 1.)
            y = F.conv2d(y, psf, groups=C, padding="same")
            # Pool the convolved flux to observation grid
            y = F.avg_pool2d(y, kernel_size=psf_super_sampling_factor, divisor_override=1 if sum_pool else None)
            # Crop out the zero padding to remove edge effects from the convolution
            pi, pj = wcs_list[i].pixel_shape
            zp = zero_padding
            y = y[..., zp:pi+zp, zp:pj+zp]
            ys.append(y)
        return torch.concat(ys, dim=1)
    return A

obs_pixels = 35 # Number of pixels in the observation
obs_pixel_size = 0.2 # Size of each pixel in the observation
model_pixels = 64 # Number of pixels in the model
model_pixel_size = 0.11 # Size of each pixel in the model
super_sampling_factor = 2 # Factor by which the PSF is super sampled.
shift_east = 0 # Pixel shift east
shift_north = 0 # Pixel shift north
wcs_angle = 0 # Orientation of the observation East of North (deg)
model_angle = None # Orientation of the model East of North (deg)
zero_padding = 0 # Zero padding in the forward model. Default is no zero-padding

center = SkyCoord(ra=10*units.deg, dec=20*units.deg)
w = make_wcs(center, 0., obs_pixels, pixel_size=obs_pixel_size * units.arcsec)
hdr = w.to_header()
# NAXIS is not populated in the to header method
hdr["NAXIS1"] = w.pixel_shape[0]
hdr["NAXIS2"] = w.pixel_shape[1]
# Option 1 (Connor's targets)
# hdr["CDELT1"] = 1
# hdr["CDELT2"] = 1
# PC = np.array([[ 2.95622685e-06,  1.33975477e-05],
                # [ 1.33618318e-05, -1.72226071e-06]])
# Option 2 (SMACS), specify CDELT and leave PC as rotation @ mirror_j
#     hdr["CDELT1"] = 0.05/3600
# hdr["CDELT2"] = 0.05 w/3600
# PC = np.array([[ 0.81783584,  0.57545159],
                # [ 0.57545159, -0.81783584]])
# Option 3 (Training WCS)
hdr["CDELT1"] = obs_pixel_size/3600
hdr["CDELT2"] = obs_pixel_size/3600
PC = np.array([[ 1, 0],
                [ 0, -1]])

hdr["PC1_1"] = PC[0, 0]
hdr["PC1_2"] = PC[0, 1]
hdr["PC2_1"] = PC[1, 0]
hdr["PC2_2"] = PC[1, 1]
w = WCS(hdr)

wcs_list = [w]
fwd_model = make_forward_model(
        gs_psf, 
        wcs_list, 
        psf_super_sampling_factor=super_sampling_factor,
        model_super_sampling_factor=model_pixels//obs_pixels,
        model_pixels=model_pixels,
        model_pixel_size=model_pixel_size * units.arcsec,
        fiducial_orientation=model_angle,
        zero_padding=zero_padding
        )

with torch.no_grad():
    # fwd_model = PhysModel(device='cuda')
    jacobian = torch.func.jacrev(lambda a: fwd_model(a[None]).squeeze(0))(torch.ones(3,64,64,device='cuda', dtype=torch.float32))
    AAT = torch.sum(jacobian**2, dim=(-3,-2,-1))[None]
# AAT = torch.sum()

class likelihood:
    def __init__(self, rubin_im, rubin_var, x_shape, sde, device='cuda'):
        batch_size, C, H, W = x_shape
        rub_im=np.repeat(rubin_im[np.newaxis, :], batch_size, axis=0)
        yim=rub_im[:,-1][:,np.newaxis] # -1 y band
        self.y_in = torch.from_numpy(yim)
        y_var=np.repeat(rubin_var[np.newaxis, :], batch_size, axis=0)
        y_var=y_var[:,-1][:,np.newaxis] # -1 y band
        self.y_var = torch.from_numpy(y_var)
        self.device = device
        self.A = fwd_model
        self.AAT = AAT
        self.sde=sde


    def LL(self,x,t):
        y= self.y_in.to(device=self.device)
        var=self.y_var.to(device=self.device)
        anneal = self.AAT * self.sde.sigma(t[0])**2
        var = var + anneal
        Ax = self.A(x)
        return ((-0.5 * (y - Ax)**2)/ var)
    
    def score(self,t,x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            log_like = self.LL(x,t).sum()
            grad_x = torch.autograd.grad(log_like, x)[0]
        return grad_x