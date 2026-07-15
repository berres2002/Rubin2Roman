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

def get_rubin_coadd(fname, wcs_json):
    coadd_rubin = np.load(fname)
    fs = fname.split('/')[-1].split('_')[0]
    try:
        wcs_rubin = _get_rubin_wcs(fs, wcs_json)
    except:
        return 1, 1
    return coadd_rubin, wcs_rubin

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

def make_cutout(img, wcs, pos_xy=None, pos_radec=None, cutout_size=64):
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
    
    if pos_xy is not None:
        sc1=wcs.pixel_to_world(pos_xy[0], pos_xy[1])
    elif pos_radec is not None:
        sc1=SkyCoord(ra=pos_radec[0], dec=pos_radec[1], unit='deg')
    else:
        raise ValueError("Please specify either pixel coordinates (pos_xy) or world coordinates (pos_radec) for the cutout center.")
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
    args = init_argparse()
    os.makedirs(args.output,exist_ok=True)
    if args.make_cutouts:
        os.makedirs(os.path.join(args.output,'data'),exist_ok=True)
    if args.make_cutouts:
        annots = {'path':[], 'img':[]}
    else:
        annots = {'roman_path':[], 'roman_img':[], 'rubin_path':[], 'rubin_img':[]}
    if args.save_rubin_cutouts and args.make_cutouts:
        annots = {'path':[], 'img':[], 'rubin_cutout_path':[], 'rubin_cutout_img':[]}
    else:
        annots = {'roman_path':[], 'roman_img':[], 'rubin_path':[], 'rubin_img':[]}
    with open('/projects/bfpq/rubin2roman/code/Rubin2Roman/data_processing/delta/dir_list.pkl', 'rb') as f:
        dir_list = pickle.load(f)
        f.close()
    with open(args.roman_wcs_json_path,'r') as f:
        roman_wcs_json = json.load(f)
        f.close()
    # randomize dir_list to avoid any biases in the order of processing the data
    np.random.seed(42069) # set seed for reproducibility
    np.random.shuffle(dir_list)
    count = 0
    break_stop = False
    for dir in dir_list:
        if break_stop: break
        rubin_glob = glob.glob(os.path.join(args.rubin_img_dir, f'{dir}/*.npy'))
        with open(glob.glob(os.path.join(args.rubin_img_dir, f'{dir}/*wcs*.json'))[0], 'r') as f:
            rubin_wcs_json = json.load(f)
            f.close()
        # roman_glob = []
        for rubin_fname in rubin_glob:
            if break_stop: break
            # roman_glob.append(os.path.join(f'/work/hdd/bdsp/yse2/truth-roman/{dir}', rubin_fname.split('/')[-1]))
            roman_fname = os.path.join(args.roman_img_dir, dir, rubin_fname.split('/')[-1])
            if os.path.exists(roman_fname):
                pass
            else:
                print(f"Roman file {roman_fname} does not exist. Please check the path and try again.")
                continue
            if args.make_cutouts:
                coadd_rubin, wcs_rubin = get_rubin_coadd(rubin_fname, rubin_wcs_json)
                coadd_roman, wcs_roman = get_roman_coadd(roman_fname, roman_wcs_json)
                if isinstance(coadd_rubin, int) or isinstance(coadd_roman, int):
                    print(f"Could not get WCS for Rubin or Roman coadd for file {rubin_fname}. Skipping this file.")
                    continue
                # TODO: add in cutout making and saving here
                b, h, w = coadd_roman.shape
                rubin_b, rubin_h, rubin_w = coadd_rubin.shape
                big_array = np.zeros((b+rubin_b, h, w))
                big_array[rubin_b:]=coadd_roman
                big_array[:rubin_b],_=reproject_rubin_to_roman(coadd_rubin, wcs_rubin, wcs_roman, coadd_roman[0])
                # TODO: write cutouts centered on table sources function
                truth_json_path = 'truth_'+rubin_fname.strip('.npy').split('/')[-1]+'.json'
                truth_json_path = os.path.join(args.rubin_img_dir, dir, truth_json_path)
                if os.path.exists(truth_json_path):
                    objs = get_objects_from_json(truth_json_path)
                    for i in range(len(objs['id'])):
                        if break_stop: break
                        cutout_data, cutout_wcs = make_cutout(big_array, wcs_roman, pos_radec=(objs['ra'][i], objs['dec'][i]), cutout_size=args.cutout_size)
                        if cutout_data is not None and np.isnan(cutout_data[0].min())==False:
                            cutout_fname = f"{rubin_fname.strip('.npy').split('/')[-1]}_cut_{objs['id'][i]}.npy"
                            path = os.path.join(args.output, 'data', cutout_fname)
                            np.save(path, cutout_data.astype(np.float32))
                            annots['path'].append(path)
                            annots['img'].append(cutout_fname)
                            count +=1
                            if args.n_test is not None and count >= args.n_test:
                                break_stop = True
                                print(f"n_test limit ({args.n_test}) has been reached, stopping now.")
                                break
                        if args.save_rubin_cutouts:
                            rubin_cutout_data, rubin_cutout_wcs = make_cutout(coadd_rubin, wcs_rubin, pos_radec=(objs['ra'][i], objs['dec'][i]), cutout_size=np.ceil(args.cutout_size*(0.11/0.2)).astype(int)) # 0.2 rubin pixel scale, 0.11 roman pixel scale, so 64 pixel cutout is ~12.8 arcsec for rubin and ~7 arcsec for roman, rounding to nearest pixel
                            if rubin_cutout_data is not None and np.isnan(rubin_cutout_data.min())==False:
                                rubin_cutout_fname = f"{rubin_fname.strip('.npy').split('/')[-1]}_cut_{objs['id'][i]}_rubin.npy"
                                rubin_path = os.path.join(args.output, 'data', rubin_cutout_fname)
                                np.save(rubin_path, rubin_cutout_data.astype(np.float32))
                                annots['rubin_cutout_path'].append(rubin_path)
                                annots['rubin_cutout_img'].append(rubin_cutout_fname)
                            

                    
            else:
                annots['roman_path'].append(roman_fname)
                annots['roman_img'].append(roman_fname.split('/')[-1])
                annots['rubin_path'].append(rubin_fname)
                annots['rubin_img'].append(rubin_fname.split('/')[-1])

    annotations = pd.DataFrame(annots)
    
    ann_path =os.path.join(args.output, 'annotations.csv')
    annotations.to_csv(ann_path, index=False)
    print(f"Annotations with length {len(annotations)} saved to {ann_path}")
    # fpath = ''

