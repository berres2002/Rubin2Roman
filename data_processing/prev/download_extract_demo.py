from astropy.io import fits
import numpy as np
import s3fs
from matplotlib import pyplot as plt
from matplotlib import patches
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
# from firefly_client import FireflyClient
from astropy.nddata import Cutout2D
from itertools import product
# from reproject import reproject_interp
from io import BytesIO
import os
import pandas as pd
from tqdm import tqdm


# Open Universe Roman and Rubin Preview Paths
BUCKET_NAME = "nasa-irsa-simulations"
ROMAN_PREFIX = "openuniverse2024/roman/preview"
ROMAN_COADD_PATH = f"{ROMAN_PREFIX}/RomanWAS/images/coadds"
TRUTH_FILES_PATH = f"{ROMAN_PREFIX}/roman_rubin_cats_v1.1.2_faint"

RUBIN_PREFIX = "openuniverse2024/rubin/preview"
RUBIN_COADD_PATH = f"{RUBIN_PREFIX}/u/descdm/preview_data_step3_2877_19_w_2024_12/20240403T150003Z/deepCoadd_calexp/2877/19"

#Centers of roman data preview blocks. Do not alter.
ra_block_centers = np.array([9.76330352298415, 9.724522605135252, 9.68574158906671,
                        9.646960496603766, 9.608179349571955, 9.56939816979703,
                        9.530616979104877, 9.491835799321422, 9.453054652272561,
                        9.414273559784032, 9.375492543681393, 9.336711625789874]) * u.deg
dec_block_centers = np.array([-44.252584927082495, -44.22480733304182, -44.197029724175756,
                            -44.16925210374898, -44.14147447502621, -44.11369684127218,
                            -44.08591920575162, -44.05814157172923, -44.03036394246976,
                            -44.0025863212379, -43.974808711298394, -43.94703111591591]) * u.deg
block_size = 100 * u.arcsec # each block is 100 arcsec across


def get_block_axis(block_centers, coord, ra_or_dec):
    ra_or_dec_coord = getattr(coord, ra_or_dec)
    block_dist_array = np.absolute(block_centers - ra_or_dec_coord)
    closest_block_idx = block_dist_array.argmin()
    if (ra_or_dec_coord < block_centers.min()-block_size/2 
        or ra_or_dec_coord > block_centers.max()+block_size/2):
        raise ValueError(f"Chosen {ra_or_dec}: {ra_or_dec_coord} not covered by OpenUniverse 2024 data preview simulated Roman coadds")
    else:
        return closest_block_idx + 12 # preview covers central 12 rows 12 columns, in a grid of 36x36 blocks
    
def get_roman_coadd_fpath(coord, filter):
    col = get_block_axis(ra_block_centers, coord, 'ra')
    row = get_block_axis(dec_block_centers, coord, 'dec')
    
    # Construct the coadd filename from the chosen filter, row, and column.
    coadd_fname_root = f"prod_{filter[0]}_{col}_{row}_map.fits"
    coadd_fpath = f"{BUCKET_NAME}/{ROMAN_COADD_PATH}/{filter}/Row{row}/{coadd_fname_root}"
    return coadd_fpath, coadd_fname_root

def get_roman_coadd(coord, filter):
    # retrive fits file of block/tile from the coadd mosiac
    coadd_s3_fpath, coadd_fname_root = get_roman_coadd_fpath(coord, filter)
    coadd_s3_uri = f"s3://{coadd_s3_fpath}"
    coadd_fname = coadd_fname_root.split('.')[0]

    with fits.open(coadd_s3_uri, fsspec_kwargs={"anon": True}) as hdul:
        # retrieve science data from coadd fits
        coadd_data = hdul[0].section[0,0, :, :]  # has (2688, 2688, 15, 1) shape, with 0th layer in the cube as science image

        # make wcs using header
        coadd_wcs = wcs.WCS(hdul[0].header, naxis=2)

        return {'data': coadd_data, 'wcs': coadd_wcs}, coadd_fname
    
# API call

def download_roman_cutouts(coords,filter_roman, split_size=4, fpath=None):
    annots = {'path':[], 'img':[]}
    if fpath is not None:
        if os.path.exists(fpath): pass
        else: raise ValueError(f"Provided path {fpath} does not exist. Please provide a valid path or set path to None to save in current working directory.")
    else:
        fpath = os.getcwd()
    for coord in tqdm(coords):
        coadd_roman,coadd_fname = get_roman_coadd(coord, filter_roman)
        coadd_data = coadd_roman['data']
        if split_size is not None:
            idx=np.linspace(0,coadd_data.shape[0],split_size+1,dtype=int)
            # cutouts = []
            for i in range(split_size):
                cut1 = coadd_data[idx[i]:idx[i+1]]
                for j in range(split_size):
                    cut2 = cut1[:, idx[j]:idx[j+1]]
                    cutout_fname = f"{coadd_fname}_cut_{i}_{j}.npy"
                    path = os.path.join(fpath, 'data', cutout_fname)
                    np.save(path,cut2.astype(np.float32))
                    annots['path'].append(path)
                    annots['img'].append(cutout_fname)
        else:           
            path = os.path.join(fpath, 'data', coadd_fname+'.npy')
            np.save(path,coadd_roman['data'].astype(np.float32))
            annots['path'].append(path)
            annots['img'].append(coadd_fname+'.npy')
    return annots
    # return path



# coord = SkyCoord(ra=9.6055383, dec=-44.1895542, unit="deg")
# filter_roman = 'H158' #F184, H158, J129, K213, and Y106 are available in the data preview

if __name__ == "__main__":
    # annots = {'path':[], 'img':[]}
    fpath = '/projects/bfpq/work/aberres2/demo_roman2'
    coords=[]
    for i in range(ra_block_centers.size): #ra_block_centers.size
        coord = SkyCoord(ra=ra_block_centers[i], dec=dec_block_centers[i], unit="deg")
        coords.append(coord)
    filter_roman = 'H158' #F184, H158, J129, K213, and Y106 are available in the data preview
    annots=download_roman_cutouts(coords,filter_roman,fpath=fpath, split_size=42)
        # annots['path'].append(path)
        # annots['img'].append(path.split('/')[-1])

    annotations = pd.DataFrame(annots)
    ann_path =os.path.join(fpath, 'annotations.csv')
    annotations.to_csv(ann_path, index=False)
    print(f"Roman cutout data paths and annotations saved to {ann_path}")
    # print(f"Roman cutout data saved to {path}")
    