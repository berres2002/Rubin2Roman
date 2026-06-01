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

COLUMNS = ['cutout_id', 'psnr_Y', 'psnr_J', 'psnr_H', 'ssim_Y', 'ssim_J', 'ssim_H','roman_flux_Y','roman_flux_J','roman_flux_H','pred_flux_Y','pred_flux_J','pred_flux_H','roman_HLR_Y','roman_HLR_J','roman_HLR_H','pred_HLR_Y','pred_HLR_J','pred_HLR_H',\
        #    'lsst_flux_u','lsst_flux_g','lsst_flux_r','lsst_flux_i','lsst_flux_z','lsst_flux_y'\
           ]

def init_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test data.")
    parser.add_argument('--checkpoints_directory', type=str, required=True, help='Directory where model checkpoints are stored.')
    parser.add_argument('--test_csv', type=str, required=True, help='CSV file containing paths to test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation outputs.')
    parser.add_argument('--n_samples', type=int, default=400, help='Number of samples to generate for each test image.')
    parser.add_argument('--steps', type=int, default=1000, help='Number of diffusion steps for sampling.')
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
    paths = paths[:50]
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

        samples=model.sample(shape=[args.n_samples,3,64,64],steps=args.steps, condition=[cims])

        full_samp = samples.cpu().numpy()
        s_median = np.median(full_samp, axis=0)
        # shape is (3, 64, 64) for the median image
        out1 = normalize_unit(s_median)
        im_norm = normalize_unit(fimg[6:]) # Normalize the target image (last 3 channels)
        psnr_val = psnr(out1, im_norm)
        dd1['psnr_Y'].append(psnr_val[0])
        dd1['psnr_J'].append(psnr_val[1])
        dd1['psnr_H'].append(psnr_val[2])
        ssim_val = struct_sim(out1, im_norm)
        dd1['ssim_Y'].append(ssim_val[0])
        dd1['ssim_J'].append(ssim_val[1])
        dd1['ssim_H'].append(ssim_val[2])
        out_cens = GetCenterPeak(out1)
        im_cens = GetCenterPeak(im_norm)
        im_hlr = get_HLR(im_norm,im_cens[:,0],im_cens[:,1])
        samp_hlr = get_HLR(out1,out_cens[:,0],out_cens[:,1])
        dd1['roman_HLR_Y'].append(im_hlr[0])
        dd1['roman_HLR_J'].append(im_hlr[1])
        dd1['roman_HLR_H'].append(im_hlr[2])
        dd1['pred_HLR_Y'].append(samp_hlr[0])
        dd1['pred_HLR_J'].append(samp_hlr[1])
        dd1['pred_HLR_H'].append(samp_hlr[2])
        # TODO: Figure out radius for aperture photometry for flux measurement
        # im_flux = get_aperture_fluxes(im_norm, im_cens[:,0], im_cens[:,1], radius=5)
        # samp_flux = get_aperture_fluxes(out1, out_cens[:,0], out_cens[:,1], radius=5)
        dd1['roman_flux_Y'].append(-999)
        dd1['roman_flux_J'].append(-999)
        dd1['roman_flux_H'].append(-999)
        dd1['pred_flux_Y'].append(-999)
        dd1['pred_flux_J'].append(-999)
        dd1['pred_flux_H'].append(-999)
    data_out = pd.DataFrame(dd1)
    os.makedirs(args.output_dir, exist_ok=True)
    data_out.to_csv(f"{args.output_dir}/evaluation_results.csv", index=False)
    print(f"Saved evaluation results to {args.output_dir}/evaluation_results.csv")