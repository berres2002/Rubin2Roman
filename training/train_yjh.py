from score_models.score_model import ScoreModel
from score_models.architectures import DDPM#, NCSNpp
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
    parser.add_argument('--data_norm', type=str, default='None', choices=['None', 'Asinh'], help='Normalization method for the data. Choose "None" for no normalization or "Asinh" for Asinh normalization.')
    parser.add_argument('--checkpoints', type=int, default=100, help='Save model checkpoints every N epochs.')
    return parser.parse_args()

def main(args):
    # dataset = CustomImageDataset(annotations_file='/projects/bfpq/work/aberres2/demo_roman_rubin_fix/annotations.csv', img_dir='/projects/bfpq/work/aberres2/demo_roman_rubin_fix/data')
    if args.data_norm == 'None':
        dataset = CustomImageDatasetYJH(annotations_file=args.annotations_file, img_dir=args.img_dir)
    elif args.data_norm == 'Asinh':
        dataset = CustomImageDatasetYJHAsinh(annotations_file=args.annotations_file, img_dir=args.img_dir)

    checkpoints_directory = args.checkpoints_directory
    if os.path.exists(checkpoints_directory):
        print(f"Warning: Checkpoints directory '{checkpoints_directory}' already exists. Existing checkpoints may be overwritten.")
    else:
        os.makedirs(checkpoints_directory, exist_ok=True)
    # checkpoints_directory = '/work/hdd/bfpq/aberres2/checkpoints/demo_cond_ncsnpp1'

    net = DDPM(channels=3)
    #net = NCSNpp(channels=3, condition=('Input',), condition_input_channels=6, resblock_type='ddpm')
    model = ScoreModel(model=net, sigma_min=1e-4, sigma_max=500, device="cuda") # VE SDE
    # model = ScoreModel(model=net, beta_min=1e-2, beta_max=20, device='cuda') # VP SDE
    print("Starting training...")
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
