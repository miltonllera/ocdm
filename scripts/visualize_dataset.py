import argparse
import os
import sys
import torch
import torchvision.utils as vutils
from torch.utils.data import Subset

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.pentominos import Pentominos
from src.dataset.dsprites import DSprites
from src.dataset.multi_shape import ColoredMultiShapeDataset

def main():
    parser = argparse.ArgumentParser(description="Visualize samples from a dataset and save them as a grid.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pentominos", "dsprites", "multi_pentominos", "multi_dsprites"],
        help="The dataset to visualize."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to visualize."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="plots/dataset_samples.png",
        help="Path where the output grid image will be saved."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset directory/file. Defaults to standard workspace paths."
    )
    parser.add_argument(
        "--n_shapes",
        type=int,
        default=2,
        help="Number of shapes to combine (for multi_* datasets)."
    )
    parser.add_argument(
        "--num_colors",
        type=int,
        default=10,
        help="Number of discrete colors (for multi_* datasets)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the grid in a matplotlib interactive plot explorer."
    )
    args = parser.parse_args()

    # Determine default paths if not specified
    if args.dataset_path is None:
        if "pentominos" in args.dataset:
            args.dataset_path = "data/datasets/pentominos/"
        else:
            args.dataset_path = "data/raw/dsprites/dsprite_train.npz"

    print(f"Loading base dataset '{args.dataset}' from: {args.dataset_path}")

    # Load base dataset
    if "pentominos" in args.dataset:
        base_dataset = Pentominos(path=args.dataset_path, prediction_type="unsupervised")
    else:
        base_dataset = DSprites(path=args.dataset_path, batch_type="unsupervised")

    # Wrap in multi-shape dataset if requested
    if args.dataset == "multi_pentominos" or args.dataset == "multi_dsprites":
        print(f"Wrapping with ColoredMultiShapeDataset (n_shapes={args.n_shapes}, num_colors={args.num_colors})")
        dataset = ColoredMultiShapeDataset(
            base_dataset=base_dataset,
            n_shapes=args.n_shapes,
            num_colors=args.num_colors,
            prediction_type="unsupervised",
            seed=args.seed
        )
    else:
        dataset = base_dataset

    total_len = len(dataset)
    print(f"Dataset loaded. Total length: {total_len}")

    # Sample images
    # We sample a range of indices or random indices
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_len, generator=rng)[:args.num_samples].tolist()
    
    print(f"Sampling {len(indices)} samples at indices: {indices}")
    images = []
    for idx in indices:
        sample = dataset[idx]
        img = sample[0] if isinstance(sample, tuple) else sample
        # If image is grayscale (1 channel), repeat to 3 channels for uniform grid plotting
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        images.append(img)

    # Stack images and create grid
    images_tensor = torch.stack(images, dim=0)
    # Calculate grid layout (e.g. approx square grid)
    nrow = int(len(images) ** 0.5)
    if nrow * nrow < len(images):
        nrow += 1

    grid = vutils.make_grid(images_tensor, nrow=nrow, padding=2, normalize=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving grid of shape {images_tensor.shape} to: {args.output_path}")
    vutils.save_image(grid, args.output_path)

    if args.show:
        print("Opening interactive Matplotlib plot explorer...")
        import matplotlib.pyplot as plt
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title(f"Dataset: {args.dataset} ({args.num_samples} samples)")
        plt.show()

    print("Done!")

if __name__ == "__main__":
    main()
