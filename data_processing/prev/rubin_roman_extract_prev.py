from astropy.io import fits
import numpy as np
# import s3fs
# from matplotlib import pyplot as plt
# from matplotlib import patches
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
# from firefly_client import FireflyClient
from astropy.nddata import Cutout2D
# from itertools import product
# from reproject import reproject_interp
# from io import BytesIO
import os
import pandas as pd
from tqdm import tqdm
from reproject import reproject_interp
from skimage.feature import peak_local_max
import pickle


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

def get_radec_bounds(wcs):
    max1=wcs.pixel_to_world(wcs._naxis[0], wcs._naxis[1])
    min1=wcs.pixel_to_world(0, 0)

    if max1.ra.deg > min1.ra.deg:
        ra_max, ra_min = max1.ra.deg, min1.ra.deg
    else:
        ra_max, ra_min = min1.ra.deg, max1.ra.deg
    if max1.dec.deg > min1.dec.deg:
        dec_max, dec_min = max1.dec.deg, min1.dec.deg
    else:
        dec_max, dec_min = min1.dec.deg, max1.dec.deg
    return ra_min, ra_max, dec_min, dec_max


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
    
def get_rubin_coadd_fpath(filter): 
    coadd_fname_root = f"deepCoadd_calexp_2877_19_{filter}_DC2_u_descdm_preview_data_step3_2877_19_w_2024_12_20240403T150003Z.fits"
    coadd_fpath = f"{BUCKET_NAME}/{RUBIN_COADD_PATH}/{filter}/{coadd_fname_root}"
    return coadd_fpath

def get_rubin_coadd(filter):
    coadd_s3_fpath = get_rubin_coadd_fpath(filter)

    with fits.open(f"s3://{coadd_s3_fpath}", fsspec_kwargs={"anon": True}) as hdul:
        # retrieve science data from coadd fits
        coadd_data = hdul[1].section[:,:]

        # make wcs using header
        coadd_wcs = wcs.WCS(hdul[1].header)

        return {'data': coadd_data, 'wcs': coadd_wcs}
    
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

# 1. Download Rubin ugrizy outside of everything
# 2. Download Roman cutouts YJH
# 3. Reproject Rubin to Roman WCS
# 4. Make cutouts of Rubin and Roman that are the same size and centered on the same coordinates

def download_rubin(filters):
    rubin_ims = np.zeros((len(filters), 4200, 4200))
    wcs_rubin = []
    for i,filter in enumerate(filters):
        coadd_rubin = get_rubin_coadd(filter)
        coadd_data = coadd_rubin['data']
        wcs = coadd_rubin['wcs']
        rubin_ims[i] = coadd_data
        wcs_rubin.append(wcs)
    return rubin_ims, wcs_rubin

def reproject_rubin_to_roman(rubin_ims, wcs_rubin, wcs_roman, coadd_roman):
    return reproject_interp((rubin_ims,wcs_rubin[0]),wcs_roman,shape_out=coadd_roman['data'].shape)

def save_cutouts(img,cutout_fname,fpath,split_size):
    annots = {'path':[], 'img':[]}
    idx=np.linspace(0,img.shape[1],split_size+1,dtype=int)
    for i in range(split_size):
        cut1 = img[:,idx[i]:idx[i+1]]
        for j in range(split_size):
            cut2 = cut1[:,:, idx[j]:idx[j+1]]
            if cut2[-1].max()> 1000: # check if there is any signal in the Roman H158 passband
                fname = cutout_fname+f"_cut_{i}_{j}.npy"
                path = os.path.join(fpath, 'data', fname)
                np.save(path,cut2.astype(np.float32))
                annots['path'].append(path)
                annots['img'].append(fname)
    return annots

def save_centered_cutouts(img, cutout_size=64, band_idx=None, fpath=None, cutout_fname=''):
    annots = {'path':[], 'img':[]}
    if img.ndim == 3:
        if band_idx is not None:
            multiband = True
            image = img[band_idx] # use specified band for finding peaks to center cutouts on
        else:
            raise ValueError("For 3D image cubes, please specify the band index to use for finding local peaks to center cutouts on.")
    elif img.ndim == 2:
        multiband = False
        image = img
    else:
        raise ValueError("Input image must be either a 2D array or a 3D cube with shape (bands, height, width)")
    coordinates = peak_local_max(
            image,
            min_distance=20,
            threshold_abs=200,
            num_peaks=100,
        )
    x,y = coordinates[:, 1], coordinates[:, 0]
    mask = (x > cutout_size) & (x < image.shape[1]-cutout_size) & (y > cutout_size) & (y < image.shape[0]-cutout_size)
    coods_x  = x[mask]
    coods_y = y[mask]
    for x,y in zip(coods_x, coods_y):
        cutout = Cutout2D(image, (x,y), (cutout_size, cutout_size)).slices_original
        if multiband:
            cutout_data = img[:, cutout[0], cutout[1]]
        else:
            cutout_data = img[cutout[0], cutout[1]]
        # save with fpath also
        if fpath is not None:
            if os.path.exists(fpath):
                fname = cutout_fname+f"_cut_{x}_{y}.npy"
                save_path = os.path.join(fpath, 'data', fname)

                np.save(save_path, cutout_data.astype(np.float32))
                annots['path'].append(save_path)
                annots['img'].append(fname)
            else: 
                raise ValueError(f"Provided path {fpath} does not exist. Please provide a valid path or set path to None to save in current working directory.")
    return annots

def save_centered_cutouts_fromTable(table, img, wcs, cutout_size=64, fpath=None, cutout_fname='',c_annots=None):
    annots = {'path':[], 'img':[], 'ra':[], 'dec':[]}
    if img.ndim == 3:
        multiband = True
        #     image = img[band_idx] # use specified band for finding peaks to center cutouts on
        # else:
        #     raise ValueError("For 3D image cubes, please specify the band index to use for finding local peaks to center cutouts on.")
    elif img.ndim == 2:
        multiband = False
        image = img
    else:
        raise ValueError("Input image must be either a 2D array or a 3D cube with shape (bands, height, width)")
    ra_min, ra_max, dec_min, dec_max = get_radec_bounds(wcs)
    tqi = table.query(f"ra > {ra_min} and ra < {ra_max} and dec > {dec_min} and dec < {dec_max}")
    for row in tqi[['ra','dec']].to_numpy():
        if c_annots is not None:
            if row[0] in c_annots['ra'] and row[1] in c_annots['dec']:
                print(f"Source at ra={row[0]}, dec={row[1]} already has a cutout, skipping this source.")
                continue
        try:
            cutout = Cutout2D(img[-1], SkyCoord(ra=row[0], dec=row[1],unit='deg'), (cutout_size, cutout_size), wcs=wcs, mode='strict').slices_original
        except:
            print(f"Could not make cutout for source at ra={row[0]}, dec={row[1]}. Skipping this source.")
            continue
        # save cutout
        if multiband:
            cutout_data = img[:, cutout[0], cutout[1]]
        else:
            cutout_data = img[cutout[0], cutout[1]]
        if fpath is not None:
            if os.path.exists(fpath):
                if row[1]<0:
                    dec_str = f"{row[1]*-1:0.4f}"
                else:
                    dec_str = f"{row[1]:0.4f}"
                fname = cutout_fname+f"_cut_{row[0]:0.4f}_{dec_str}.npy"
                save_path = os.path.join(fpath, 'data', fname)

                np.save(save_path, cutout_data.astype(np.float32))
                annots['path'].append(save_path)
                annots['img'].append(fname)
                annots['ra'].append(row[0])
                annots['dec'].append(row[1])
            else: 
                raise ValueError(f"Provided path {fpath} does not exist. Please provide a valid path or set path to None to save in current working directory.")
    return annots

def download_roman(coords, filter_roman, rubin_ims, wcs_rubin, fpath=None, split_size=4, max_images=2200, table=None):
    # roman_ims = []
    # wcs_roman = []
    #allocate big array
    annots = {'path':[], 'img':[], 'ra':[], 'dec':[]}
    nogals = False
    for coord in tqdm(coords):
        big_array = np.zeros((9 , 2688, 2688))
        for i,filter in enumerate(filter_roman):
            coadd_roman,coadd_fname = get_roman_coadd(coord, filter)
            coadd_data = coadd_roman['data']
            roman_wcs = coadd_roman['wcs']
            # annoying query for testing
        #     ra_min, ra_max, dec_min, dec_max = get_radec_bounds(roman_wcs)
        #     tqi = table.query(f"ra > {ra_min} and ra < {ra_max} and dec > {dec_min} and dec < {dec_max}")
        #     if len(tqi)==0:
        #         # print(f"No sources from the catalog found in Roman cutout centered at ra={coord.ra.deg}, dec={coord.dec.deg}. Skipping this cutout.")
        #         nogals = True
        #         break
        #     # end annoying query
            big_array[i-3]=coadd_data
        # #annoying check here
        # if nogals:
        #     nogals = False
        #     continue
        # #end annoying check
        # rubin shape (6, 2688, 2688)
        rubin_reprojected,_ = reproject_rubin_to_roman(rubin_ims, wcs_rubin, roman_wcs, coadd_roman)
        if np.isnan(rubin_reprojected[0].min()) ==False:
            big_array[:6]=rubin_reprojected
            name_split = coadd_fname.split('_')
            cutout_fname = f"ugrizy_YJH_{name_split[2]}_{name_split[3]}_map"
            # ans = save_cutouts(big_array, cutout_fname, fpath, split_size=split_size)
            # ans = save_centered_cutouts(big_array, cutout_size=64, band_idx=-1, fpath=fpath, cutout_fname=cutout_fname)
            ans = save_centered_cutouts_fromTable(table, big_array, roman_wcs, cutout_size=64, fpath=fpath, cutout_fname=cutout_fname, c_annots=annots)
            annots['path'].extend(ans['path'])
            annots['img'].extend(ans['img'])
            annots['ra'].extend(ans['ra'])
            annots['dec'].extend(ans['dec'])
        if len(annots['path']) > max_images and max_images>0: # stop after we have downloaded a certain number of cutouts to avoid memory issues and set max_images to -1 to download all cutouts
            print(f"Reached max number of images N = {max_images}, stopping download.")
            break
    return annots


# coord = SkyCoord(ra=9.6055383, dec=-44.1895542, unit="deg")
# filter_roman = 'H158' #F184, H158, J129, K213, and Y106 are available in the data preview

if __name__ == "__main__":
    # annots = {'path':[], 'img':[]}
    fpath = '/work/hdd/bfpq/aberres2/brightest_gals_cutouts_64'
    # fpath = '/Users/aberres/Desktop/research/rubin2roman/data/table_demo'
    # coords=[]
    x,y=np.meshgrid(ra_block_centers[2:-2], dec_block_centers[2:-2]) # only use the full 8 by 8 Roman blocks that cover the full Rubin preview area
    coords = SkyCoord(ra=x, dec=y, unit="deg")
    coords = coords.flatten()
    # for i in range(ra_block_centers.size): #ra_block_centers.size
    #     coord = SkyCoord(ra=ra_block_centers[i], dec=dec_block_centers[i], unit="deg")
    #     coords.append(coord)
    # table = pd.read_csv("roman_rubin_cat_v1.1.2_faint.csv") # read in the Roman Rubin catalog to use for centering cutouts on real sources
    # table = pd.read_csv('/Users/aberres/Desktop/research/rubin2roman/test_galaxy_cat.csv') # smaller table with only 2 sources for testing
    df = pd.read_parquet('/projects/bfpq/rubin2roman/galaxy_10307.parquet')
    with open('/projects/bfpq/rubin2roman/bright_20000_y_flux_gals_inds.pkl', 'rb') as f:
        inds = pickle.load(f)
        f.close()
    table = df.iloc[inds].reset_index(drop=True) # smaller table with only the brightest 20000 sources in the Rubin preview
    del(inds)
    del(df)
    filter_roman = ['Y106','J129','H158'] #F184, H158, J129, K213, and Y106 are available in the data preview
    filter_rubin = ['u','g','r','i','z','y']
    print("Downloading Rubin coadds...")
    rubin_ims, wcs_rubin = download_rubin(filter_rubin)
    print("Rubin coadds downloaded!\nDownloading/Saving Roman cutouts...")
    annots = download_roman(coords, filter_roman, rubin_ims, wcs_rubin, fpath=fpath, max_images=-1,table=table)
    # annots=download_roman_cutouts(coords,filter_roman,fpath=fpath, split_size=42)
        # annots['path'].append(path)
        # annots['img'].append(path.split('/')[-1])

    annotations = pd.DataFrame(annots)
    ann_path =os.path.join(fpath, 'annotations.csv')
    annotations.to_csv(ann_path, index=False)
    print("Saved",len(annotations), "images")
    print(f"Roman and Rubin cutout data paths and annotations saved to {ann_path}")
    # print(f"Roman cutout data saved to {path}")
    