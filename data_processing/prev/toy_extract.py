import numpy as np
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import os
import pandas as pd
from tqdm import tqdm
from reproject import reproject_interp
from skimage.feature import peak_local_max
import pickle
import glob
import json
from datetime import datetime
from astropy.io import fits

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
        coadd_data = hdul[3].section[:,:]

        # make wcs using header
        coadd_wcs = wcs.WCS(hdul[3].header)

        return {'data': coadd_data, 'wcs': coadd_wcs}

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

def _get_rubin_wcs(key_name, wcs_json):
    return wcs.WCS(wcs_json[key_name])

def _get_roman_wcs(fname, wcs_json):
    return wcs.WCS(wcs_json[fname]['wcs'])

def get_roman_coadd(fname, wcs_json):
    # open the file
    coadd_roman = np.load(fname)
    fs = fname.split('/')
    wcs_entry = 'roman_data/truth/'+fs[-2]+'/'+fs[-1]
    try:
        wcs_roman = _get_roman_wcs(wcs_entry, wcs_json)
    except:
        return 1, 1
    return coadd_roman, wcs_roman

# def get_rubin_coadd(fname, wcs_json):
#     coadd_rubin = np.load(fname)
#     fs = fname.split('/')[-1].split('_')[0]
#     try:
#         wcs_rubin = _get_rubin_wcs(fs, wcs_json)
#     except:
#         return 1, 1
#     return coadd_rubin, wcs_rubin

def reproject_rubin_to_roman(rubin_ims, wcs_rubin, wcs_roman, coadd_roman):
    return reproject_interp((rubin_ims,wcs_rubin),wcs_roman,shape_out=coadd_roman.shape)

def get_objects_from_json(path):
    objs = {'id':[], 'ra':[], 'dec':[]}
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    for i in range(len(data)):
        # truth_type 2 is star, 1 is galaxy
        if data[i]['truth_type'] == 1:
            objs['id'].append(data[i]['id'])
            objs['ra'].append(data[i]['ra'])
            objs['dec'].append(data[i]['dec'])
    return objs

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

def make_cutout(img, wcs, coord, cutout_size=64):
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
    
    sc1=coord
    try:
        cutout = Cutout2D(img[0], sc1, (cutout_size, cutout_size), wcs=wcs, mode='strict')
        cutout_slices = cutout.slices_original
        cutout_wcs = cutout.wcs
    except:
        print(f"Could not make cutout for source at ra={sc1.ra.deg}, dec={sc1.dec.deg}. Skipping this source.")
        return None, None
    if multiband:
        cutout_data = img[:, cutout_slices[0], cutout_slices[1]]
    else:
        cutout_data = img[cutout_slices[0], cutout_slices[1]]

    return cutout_data, cutout_wcs

def init_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Script to extract cutouts from Rubin and Roman coadds for training and evaluation of Rubin-to-Roman image translation models.")
    parser.add_argument('--make_cutouts', action=argparse.BooleanOptionalAction, help='Whether to make cutouts from the coadd images. If False, the script will only add the paths to the full coadd images to the annotations file without making cutouts.')
    parser.add_argument('--save_rubin_cutouts', action=argparse.BooleanOptionalAction, help='Whether to save the not-reprojected Rubin cutouts along with the reprojected Rubin cutouts.')
    parser.add_argument('--cutout_size', type=int, default=64, help='Size of the square cutouts to extract (in pixels).')
    parser.add_argument('--rubin_img_dir', type=str, default='/work/hdd/bdsp/yse2/lsst_data/truth', help='Directory containing the Rubin coadd .npy files organized in subdirectories by tract and patch.')
    parser.add_argument('--roman_img_dir', type=str, default='/work/hdd/bdsp/yse2/truth-roman', help='Directory containing the Roman coadd .npy files organized in subdirectories by tract and patch.')
    parser.add_argument('--output', type=str, help='Directory where the extracted cutouts and annotations will be saved.')
    # parser.add_argument('--dir_list_path', type=str, default='dir_list.pkl', help='Path to the pickle file containing the list of directories to process.')
    parser.add_argument('--roman_wcs_json_path', type=str, default='/projects/bfhm/yse2/annotations_roman/all_wcs.json', help='Path to the JSON file containing WCS information for the Roman data.')
    parser.add_argument('--n_test',type=int,default=None)
    return parser.parse_args()

if __name__ == "__main__":
    t1 = datetime.now()
    args = init_argparse()
    os.makedirs(args.output,exist_ok=True)
    if args.make_cutouts:
        os.makedirs(os.path.join(args.output,'data'),exist_ok=True)
    ann_path =os.path.join(args.output, 'test1_rubin_annotations.csv')
    df = pd.read_csv(ann_path)
    annots = {'var_path':[]}
    filter_rubin = ['u','g','r','i','z','y']
    print("Downloading Rubin coadds...")
    rubin_ims, wcs_rubin = download_rubin(filter_rubin)
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        ra, dec = row['ra'], row['dec']
        sc1 = SkyCoord(ra=ra, dec=dec, unit='deg')
        cutout,_ = make_cutout(rubin_ims, wcs_rubin[0], coord=sc1, cutout_size=32)
        np.save(os.path.join(args.output,'data',f"rubin_ugrizy_var_{ra:0.4f}_{dec:0.4f}.npy"), cutout)
        annots['var_path'].append(os.path.join(args.output,'data',f"rubin_ugrizy_var_{ra:0.4f}_{dec:0.4f}.npy"))
        # annots['img'].append(f"rubin_ugrizy_{ra:0.4f}_{dec:0.4f}.npy")
        # annots['ra'].append(ra)
        # annots['dec'].append(dec)
    annotations = pd.DataFrame(annots)

    # Joining to existing annotations file
    
    # df_existing = pd.read_csv(ann_path)
    df2=df.join(annotations)
    df2.to_csv(ann_path, index=False)
    t2 = datetime.now()
    print(f"Annotations with length {len(annotations)} saved to {ann_path}")
    print(f"Total time taken: {t2-t1}")
    # fpath = ''

