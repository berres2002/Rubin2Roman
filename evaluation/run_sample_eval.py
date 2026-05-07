from score_models.score_model import ScoreModel
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
import pandas as pd
import torch
from tqdm import tqdm
from eval_code.eval import *

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
    checkpoints_directory = args.checkpoints_directory
    model = ScoreModel(checkpoints_directory=checkpoints_directory)

    # df = pd.read_csv('/work/hdd/bfpq/aberres2/brightest_gals_cutouts_64/test1.csv')
    df = pd.read_csv(args.test_csv)

    paths= df['path'].values
    # randomize the order of the paths
    np.random.seed(42069)
    np.random.shuffle(paths)

    #TEST CASE
    paths = paths[:1]

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = np.load(path)
        fimg = ZScoreNormalize(img)
        nimg = fimg[:6] # First 6 channels as the conditioning image
        # duplicate to n_samples batches
        nimg = np.repeat(nimg[np.newaxis, :], args.n_samples, axis=0)
        assert nimg.ndims == 4
        cims = torch.from_numpy(nimg)
        # cims = cims.unsqueeze(0)
        cims = cims.to(model.device)

        samples=model.sample(shape=[400,3,64,64],steps=args.steps, condition=[cims])

        full_samp = samples.cpu().numpy()
        s_median = np.median(full_samp, axis=0)
        # shape is (3, 64, 64) for the median image
        out1 = normalize_unit(s_median)
        im_norm = normalize_unit(img[6:]) # Normalize the target image (last 3 channels)