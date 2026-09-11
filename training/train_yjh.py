from score_models.score_model import ScoreModel
from score_models.architectures import DDPM, NCSNpp
# from cond_dataset_demo import CustomImageDataset
from pytorch_dataset import CustomImageDatasetYJH, CustomImageDatasetYJHAsinh
import pickle
import os
import torch.multiprocessing as mp
import argparse

def init_argparse():
    parser = argparse.ArgumentParser(description="Train the model on the dataset.")
    parser.add_argument('--annotations_file', type=str, required=True, help='CSV file containing paths to training images.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing training images.')
    parser.add_argument('--checkpoints_directory', type=str, required=True, help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--data_norm', type=str, default='None', choices=['None', 'Asinh', 'mult_val'], help='Normalization method for the data. Choose "None" for no normalization or "Asinh" for Asinh normalization.')
    parser.add_argument('--sigma_min', type=float, default=None, help='Minimum sigma value for the VE SDE.')
    parser.add_argument('--sigma_max', type=float, default=None, help='Maximum sigma value for the VE SDE.')
    parser.add_argument('--beta_min', type=float, default=None, help='Minimum beta value for the VP SDE.')
    parser.add_argument('--beta_max', type=float, default=None, help='Maximum beta value for the VP SDE.')
    parser.add_argument('--checkpoints', type=int, default=100, help='Save model checkpoints every N epochs.')
    return parser.parse_args()

def main(args):
    # dataset = CustomImageDataset(annotations_file='/projects/bfpq/work/aberres2/demo_roman_rubin_fix/annotations.csv', img_dir='/projects/bfpq/work/aberres2/demo_roman_rubin_fix/data')
    if args.data_norm == 'None':
        dataset = CustomImageDatasetYJH(annotations_file=args.annotations_file, img_dir=args.img_dir)
    elif args.data_norm == 'Asinh':
        dataset = CustomImageDatasetYJHAsinh(annotations_file=args.annotations_file, img_dir=args.img_dir)
    elif args.data_norm == 'mult_val':
        mult_val = 1/159.23617710583153
        dataset = CustomImageDatasetYJH(annotations_file=args.annotations_file, img_dir=args.img_dir, mult_val=mult_val)

    checkpoints_directory = args.checkpoints_directory
    if os.path.exists(checkpoints_directory):
        print(f"Warning: Checkpoints directory '{checkpoints_directory}' already exists. Existing checkpoints may be overwritten.")
    else:
        os.makedirs(checkpoints_directory, exist_ok=True)
    # checkpoints_directory = '/work/hdd/bfpq/aberres2/checkpoints/demo_cond_ncsnpp1'
    if args.sigma_min is not None and args.sigma_max is not None:
        net = NCSNpp(channels=3)
        model = ScoreModel(model=net, sigma_min=args.sigma_min, sigma_max=args.sigma_max, device="cuda") # VE SDE
        print(f"Using VE SDE with sigma_min={args.sigma_min} and sigma_max={args.sigma_max}.")
    elif args.beta_min is not None and args.beta_max is not None:
        net = DDPM(channels=3)
        model = ScoreModel(model=net, beta_min=args.beta_min, beta_max=args.beta_max, device='cuda') # VP SDE
        print(f"Using VP SDE with beta_min={args.beta_min} and beta_max={args.beta_max}.")
    else:
        raise ValueError("You must specify either sigma_min and sigma_max for VE SDE or beta_min and beta_max for VP SDE.")
    # net = DDPM(channels=3)
    # net = NCSNpp(channels=3)
    #net = NCSNpp(channels=3, condition=('Input',), condition_input_channels=6, resblock_type='ddpm')
    # model = ScoreModel(model=net, sigma_min=1e-4, sigma_max=500, device="cuda") # VE SDE
    # model = ScoreModel(model=net, beta_min=1e-2, beta_max=20, device='cuda') # VP SDE
    print(f"Starting training with {args.data_norm} data normalization...")
    # 200 * 4 = 800, 400 * 5 = 2000
    # 1000 epochs with batch size 64

    loss=model.fit(dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, checkpoints_directory = checkpoints_directory, shuffle=True, num_workers = args.num_workers, checkpoints = args.checkpoints)
    with open(os.path.join(checkpoints_directory, 'training_loss.pkl'), 'wb') as f:
        pickle.dump(loss, f)
        f.close()
    print("Training complete. Model checkpoints saved in:", checkpoints_directory)

if __name__ == "__main__":
    args = init_argparse()
    mp.set_start_method("spawn", force=True)
    main(args)
