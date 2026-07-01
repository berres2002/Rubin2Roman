from score_models.score_model import ScoreModel
#import matplotlib.pyplot as plt
import numpy as np
#from astropy.visualization import ZScaleInterval
import pandas as pd
import torch
from tqdm import tqdm
from eval import *
import os

# def _ZScoreNormalize(image: np.ndarray) -> np.ndarray:
#     """
#     Z-score normalize each channel of the input image independently.
    
#     Parameters:
#     image (np.ndarray): Input image of shape (C, H, W) where C is the number of channels.
    
#     Returns:
#     np.ndarray: Z-score normalized image of the same shape as the input.
#     """
#     mean = np.mean(image, axis=(1, 2), keepdims=True)
#     std = np.std(image, axis=(1, 2), keepdims=True)
#     return (image - mean) / std
#cols = ['semimajor_axis','semiminor_axis','orientation','ellipticity','kron_flux']
def columns_constructor():
    bands = ['Y', 'J', 'H']
    comps = ['psnr','ssim']
    mfeat = ['flux','HLR','semimajor_axis','semiminor_axis','orientation','ellipticity','kron_flux']
    start = ['cutout_id']
    for i in range(len(bands)):
        for j in range(len(comps)):
            start.append(f"{comps[j]}_{bands[i]}")
        for jj in range(len(mfeat)):
            start.append(f"roman_{mfeat[jj]}_{bands[i]}")
            start.append(f"pred_{mfeat[jj]}_{bands[i]}")
    return start
# COLUMNS = ['cutout_id', 'psnr_Y', 'psnr_J', 'psnr_H', 'ssim_Y', 'ssim_J', 'ssim_H','roman_flux_Y','roman_flux_J','roman_flux_H','pred_flux_Y','pred_flux_J','pred_flux_H','roman_HLR_Y','roman_HLR_J','roman_HLR_H','pred_HLR_Y','pred_HLR_J','pred_HLR_H',\
        #    'roman_a_Y', 'roman_a_J', 'roman_a_H', 'roman_b_Y', 'roman_b_J', 'roman_b_H',
        #    'lsst_flux_u','lsst_flux_g','lsst_flux_r','lsst_flux_i','lsst_flux_z','lsst_flux_y'\
        #    ]
COLUMNS = columns_constructor()

def init_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test data.")
    parser.add_argument('--checkpoints_directory', type=str, required=True, help='Directory where model checkpoints are stored.')
    parser.add_argument('--test_csv', type=str, required=True, help='CSV file containing paths to test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation outputs.')
    parser.add_argument('--n_samples', type=int, default=400, help='Number of samples to generate for each test image.')
    parser.add_argument('--steps', type=int, default=1000, help='Number of diffusion steps for sampling.')
    parser.add_argument('--n_test',type=int,default=50,help='Number of test galaxies to use. Can use up to the full size of the test set.')
    return parser.parse_args()

if __name__ == "__main__":
    args = init_argparse()
    # checkpoints_directory = '/work/hdd/bfpq/aberres2/checkpoints/demo_cond_10k'
    dd1 = {col:[] for col in COLUMNS}
    checkpoints_directory = args.checkpoints_directory
    print('Loading model...')
    model = ScoreModel(checkpoints_directory=checkpoints_directory)

    # df = pd.read_csv('/work/hdd/bfpq/aberres2/brightest_gals_cutouts_64/test1.csv')
    print('Loading test data paths...')
    df = pd.read_csv(args.test_csv)

    paths= np.array(df['path'].values)
    # img_names = df['img'].values
    # randomize the order of the paths
    np.random.seed(42069)
    np.random.shuffle(paths)

    #TEST CASE
    if args.n_test < paths.size:
        paths = paths[:args.n_test]
    fail_counter = 0
    print('Evaluating model on test data...')
    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = np.load(path)
        name = path.split('/')[-1].strip('.npy')
        dd1['cutout_id'].append(name)
        fimg = ZScoreNormalize(img)
        nimg = fimg[:6] # First 6 channels as the conditioning image
        # duplicate to n_samples batches
        nimg = np.repeat(nimg[np.newaxis, :], args.n_samples, axis=0)
        assert nimg.ndim == 4
        cims = torch.from_numpy(nimg)
        # cims = cims.unsqueeze(0)
        cims = cims.to(model.device)

        samples=model.sample(shape=[args.n_samples,3,64,64],steps=args.steps, condition=[cims], verbose=0)

        full_samp = samples.cpu().numpy()
        s_median = np.median(full_samp, axis=0)
        if s_median.max() == 0.0 or np.isnan(s_median.max()) or np.sum(s_median) == 0.0 or np.isnan(np.sum(s_median)):
            print('Model produced empty or nan valued images')
            continue
        # shape is (3, 64, 64) for the median image
        # out1 = normalize_unit(s_median)
        # im_norm = normalize_unit(fimg[6:]) # Normalize the target image (last 3 channels)
        # Trying to use center normalization instead of unit normalization for both the predicted and target images
        # out1 = normalize_center(s_median)
        # im_norm = normalize_center(fimg[6:]) # Normalize the target image (last 3 channels)
        out1 = s_median
        if abs(out1).max() == np.inf or abs(im_norm).max() == np.inf:
            print('Normalization produced inf values. Skipping.')
            continue
        try:
            out_cens = GetCenterPeak(out1)
        except:
            print('Peak finding algorithm failed for predicted source. Skipping...')
            fail_counter+=1
            save_path = f"/work/hdd/bfpq/aberres2/failed_eval_ims/test1/pred_fail{fail_counter}.npy"
            print(f'Saving images as {save_path}')
            np.save(save_path,np.vstack((out1,im_norm)))
            continue
        try:
            im_cens = GetCenterPeak(im_norm)
        except:
            print('Peak finding algorithm failed for Roman source. Skipping...')
            fail_counter+=1
            save_path = f"/work/hdd/bfpq/aberres2/failed_eval_ims/test1/roman_fail{fail_counter}.npy"
            print(f'Saving images as {save_path}')
            np.save(save_path,np.vstack((out1,im_norm)))
            continue
        pred_morph = get_morph(out1)
        roman_morph = get_morph(im_norm)
        cols = pred_morph.keys()
        bands = ['Y','J','H']
        for j in range(len(bands)):
            for k in range(len(cols)):
                dd1[f'pred_{cols[k]}_{bands[j]}'].append(np.float64(pred_morph[j][cols[k]]))
                dd1[f'roman_{cols[k]}_{bands[j]}'].append(np.float64(roman_morph[j][cols[k]]))
        out1 = normalize_unit(s_median)
        im_norm = normalize_unit(fimg[6:]) # Normalize the target image (last 3 channels)
        # save some of the model outputs REMOVE WHEN NOT TESTING
        # np.save(f'/work/hdd/bfpq/aberres2/test_eval_imgs/pred_{i}.npy',out1)
        # np.save(f'/work/hdd/bfpq/aberres2/test_eval_imgs/roman_{i}.npy',im_norm)
        psnr_val = psnr(out1, im_norm)
        dd1['psnr_Y'].append(psnr_val[0])
        dd1['psnr_J'].append(psnr_val[1])
        dd1['psnr_H'].append(psnr_val[2])
        ssim_val = struct_sim(out1, im_norm)
        dd1['ssim_Y'].append(ssim_val[0])
        dd1['ssim_J'].append(ssim_val[1])
        dd1['ssim_H'].append(ssim_val[2])
        # peak_local_max is index coordinates so x and y are reversed
        im_hlr = get_HLR(im_norm,im_cens[:,1],im_cens[:,0])
        samp_hlr = get_HLR(out1,out_cens[:,1],out_cens[:,0])
        dd1['roman_HLR_Y'].append(im_hlr[0])
        dd1['roman_HLR_J'].append(im_hlr[1])
        dd1['roman_HLR_H'].append(im_hlr[2])
        dd1['pred_HLR_Y'].append(samp_hlr[0])
        dd1['pred_HLR_J'].append(samp_hlr[1])
        dd1['pred_HLR_H'].append(samp_hlr[2])
        # TODO: Figure out radius for aperture photometry for flux measurement
        # Choosing a radius of 10 pixels
        im_flux,im_flux_errs = get_aperture_fluxes(im_norm, im_cens[:,0], im_cens[:,1], radius=10)
        samp_flux, samp_flux_errs = get_aperture_fluxes(out1, out_cens[:,0], out_cens[:,1], radius=10)
        dd1['roman_flux_Y'].append(im_flux[0])
        dd1['roman_flux_J'].append(im_flux[1])
        dd1['roman_flux_H'].append(im_flux[2])
        dd1['pred_flux_Y'].append(samp_flux[0])
        dd1['pred_flux_J'].append(samp_flux[1])
        dd1['pred_flux_H'].append(samp_flux[2])
        
    data_out = pd.DataFrame(dd1)
    os.makedirs(args.output_dir, exist_ok=True)
    data_out.to_csv(f"{args.output_dir}/evaluation_results.csv", index=False)
    print(f"Saved evaluation results to {args.output_dir}/evaluation_results.csv")