from score_models.score_model import ScoreModel
#import matplotlib.pyplot as plt
import numpy as np
#from astropy.visualization import ZScaleInterval
import pandas as pd
import torch
from tqdm import tqdm
from eval import *
import os
from datetime import datetime
from likelihood import likelihood
import h5py
import json

def init_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Sample the trained model on test data.")
    parser.add_argument('--checkpoints_directory', type=str, required=True, help='Directory where model checkpoints are stored.')
    parser.add_argument('--model_type', type=str, required=True, choices=['condition','likelihood'], help='Type of model to evaluate: "condition" for conditional model, "likelihood" for likelihood sampling model.')
    parser.add_argument('--test_csv', type=str, required=True, help='CSV file containing paths to test images.')
    parser.add_argument('--test_rubin_csv', type=str,default=None,help='CSV file containing Rubin catalog data for test images. If not provided, Rubin catalog features will not be evaluated.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save sampled images.')
    parser.add_argument('--n_samples', type=int, default=400, help='Number of samples to generate for each test image.')
    parser.add_argument('--steps', type=int, default=1000, help='Number of diffusion steps for sampling.')
    parser.add_argument('--n_test',type=int,default=50,help='Number of test galaxies to use. Can use up to the full size of the test set.')
    parser.add_argument('--single_band', action=argparse.BooleanOptionalAction, help='If set, evaluate only on the Y band. Otherwise, evaluate on YJH bands. Does not work on conditional model.')
    parser.add_argument('--seed', type=int, default=42069, help='Random seed for reproducibility.')
    return parser.parse_args()

def sample(args, sample_dset, true_dset):
    checkpoints_directory = args.checkpoints_directory
    print('Loading model...')
    model = ScoreModel(checkpoints_directory=checkpoints_directory)

    # df = pd.read_csv('/work/hdd/bfpq/aberres2/brightest_gals_cutouts_64/test1.csv')
    print('Loading test data paths...')
    df = pd.read_csv(args.test_csv)
    df_idx = df.index.values.copy()
    paths= np.array(df['path'].values)

    if args.test_rubin_csv is not None:
        rubin_df = pd.read_csv(args.test_rubin_csv)
        rubin_paths = np.array(rubin_df['path'].values)
        rubin_var_paths = np.array(rubin_df['var_path'].values)
    elif 'rubin_path' in df.columns and 'var_path' in df.columns:
        rubin_paths = np.array(df['rubin_path'].values)
        rubin_var_paths = np.array(df['var_path'].values)
    else:
        rubin_paths = None
        rubin_var_paths = None
        print("No Rubin catalog data provided. Rubin catalog features will not be evaluated.")
    # img_names = df['img'].values
    # randomize the order of the paths
    np.random.seed(args.seed)
    np.random.shuffle(df_idx)

    #TEST CASE
    if args.n_test < df_idx.size:
        df_idx = df_idx[:args.n_test]
    # fail_counter = 0
    print('Evaluating model on test data...')
    for i in tqdm(range(len(df_idx))):
        idx = df_idx[i]
        path = paths[idx]
        img = np.load(path)
        # name = path.split('/')[-1].strip('.npy')
        
        if args.model_type == 'condition':
            # fimg = ZScoreNormalize(img)
            nimg = AsinhNormalize(img[:6]) # First 6 channels as the conditioning image
            # duplicate to n_samples batches
            nimg = np.repeat(nimg[np.newaxis, :], args.n_samples, axis=0)
            assert nimg.ndim == 4
            cims = torch.from_numpy(nimg)
            # cims = cims.unsqueeze(0)
            cims = cims.to(model.device)

            samples=model.sample(shape=[args.n_samples,3,64,64],steps=args.steps, condition=[cims], verbose=0)

            full_samp = samples.cpu().numpy()
            full_samp_nn = AsinhReverseNormalize(full_samp)
        elif args.model_type == 'likelihood':
            # mult_val = 1/159.23617710583153
            rubin_img = np.load(rubin_paths[idx])
            rubin_var = np.load(rubin_var_paths[idx])
            if args.single_band:
                lf1 = likelihood(rubin_img, rubin_var, (args.n_samples,1,64,64), sde=model.sde, device=model.device)
                out_shape = [args.n_samples,1,64,64]
            else:
                lf1 = likelihood(rubin_img, rubin_var, (args.n_samples,3,64,64), sde=model.sde, device=model.device)
                out_shape = [args.n_samples,3,64,64]

            samples=model.sample(shape=out_shape,steps=args.steps, likelihood_score_fn=lf1.score,guidance_factor=1, verbose=0)
            full_samp_nn = samples.cpu().numpy()#/mult_val
        s_median = np.median(full_samp_nn, axis=0)
        if s_median.max() == 0.0 or np.isnan(s_median.max()) or np.sum(s_median) == 0.0 or np.isnan(np.sum(s_median)):
            print('Model produced empty or nan valued images')
            continue
        if args.single_band:
            img = img[0,np.newaxis] # Add a new axis to make it (1, 64, 64) for single band
        timg = (img/img.sum(axis=(1,2), keepdims=True))*(s_median.sum(axis=(1,2),keepdims=True))
        sample_dset[i] = s_median
        true_dset[i] = timg

def main(args, h5_filename):
    h5_path = os.path.join(args.output_dir, h5_filename)
    if os.path.exists(h5_path):
        print(f"Warning: HDF5 file '{h5_path}' already exists. Existing data may be overwritten.")
        raise FileExistsError(f"HDF5 file '{h5_path}' already exists. Please choose a different filename or remove the existing file.")
    else:
        with h5py.File(h5_path, 'w') as h5_file:
            if args.single_band:
                sample_dset = h5_file.create_dataset('samples', shape=(args.n_test, 1, 64, 64), dtype=np.float32)
                true_dset = h5_file.create_dataset('true_images', shape=(args.n_test, 1, 64, 64), dtype=np.float32)
            else:
                sample_dset = h5_file.create_dataset('samples', shape=(args.n_test, 3, 64, 64), dtype=np.float32)
                true_dset = h5_file.create_dataset('true_images', shape=(args.n_test, 3, 64, 64), dtype=np.float32)
            sample(args, sample_dset, true_dset)
            print(f"Sampling completed. Data saved to '{h5_path}'.")

if __name__ == "__main__":
    args = init_argparse()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f'commandline_args_{timestamp}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    h5_filename = f"sampled_images_{timestamp}.h5"
    main(args, h5_filename)
    