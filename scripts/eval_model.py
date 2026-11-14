import csv
import inspect
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_dataset,
    load_model,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _to_framed_pil(tensors: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(tensors.cpu(), nrow=nrow, padding=2, pad_value=0.5, normalize=False)
    grid_np = (grid.clip(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(grid_np, mode="RGB")


def evaluate_split(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.inference_mode():
        for x, _ in tqdm(loader, desc="Evaluating reconstructions"):
            x = x.to(device)
            x_hat = model.reconstruction(x)
            batch_size = x.size(0)
            total_mse += F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(dim=1).sum().item()
            total_mae += F.l1_loss(x_hat, x, reduction="none").flatten(1).mean(dim=1).sum().item()
            total_samples += batch_size

    return {
        "num_samples": total_samples,
        "mse": total_mse / max(total_samples, 1),
        "mae": total_mae / max(total_samples, 1),
    }


def plot_split_reconstructions(
    model,
    dataset,
    device: torch.device,
    num_samples: int = 8,
    seed: int = 42,
) -> tuple[Image.Image, Image.Image | None]:
    n = min(num_samples, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    x = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
    raw_model = getattr(model, "_orig_mod", model)
    supports_masks = "return_masks" in inspect.signature(raw_model.reconstruction).parameters

    with torch.inference_mode():
        if supports_masks:
            x_hat, masks = model.reconstruction(x, return_masks=True)
        else:
            x_hat = model.reconstruction(x)
            masks = None

    recons_img = _to_framed_pil(torch.cat([x[:n], x_hat[:n]]), nrow=n)

    masks_img = None
    if masks is not None:
        if masks.ndim == 4:
            # FigureGroundAutoencoder returns (B, 1, H, W); unify with (B, S, 1, H, W)
            masks = masks.unsqueeze(1)
        masks = masks[:n].expand(-1, -1, x.size(1), -1, -1)
        mask_rows = masks.transpose(0, 1).flatten(end_dim=1)
        masks_img = _to_framed_pil(torch.cat([x[:n], mask_rows]), nrow=n)

    return recons_img, masks_img


def save_metrics_csv(output_dir: Path, prefix: str, metrics_by_split: dict[str, dict[str, float]]):
    csv_path = output_dir / f"{prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Split", "Num Samples", "MSE", "MAE"])
        for split_name, metrics in metrics_by_split.items():
            writer.writerow([
                split_name,
                metrics["num_samples"],
                metrics["mse"],
                metrics["mae"],
            ])
    _logger.info(f"Metrics saved to {csv_path}")


def main(
    dataset_name: str,
    model_name: str,
    model_checkpoint: Path,
    filter_expr: str | None = None,
    output_dir: Path = Path("plots/reconstructions"),
    batch_size: int = 64,
    num_samples: int = 8,
    seed: int = 42,
    splits: list[str] = ("train", "test"),
):
    run_subdir = model_checkpoint.parts[-2] if len(model_checkpoint.parts) >= 2 else model_checkpoint.stem
    output_dir = output_dir / run_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset_name}"

    _logger.info("Loading dataset...")
    if filter_expr:
        train_dataset, test_dataset, train_loader, test_loader = load_dataset(
            dataset_name=dataset_name,
            filter_expr=filter_expr,
            batch_size=batch_size,
        )
        split_data = {
            "train": (train_dataset, train_loader),
            "test": (test_dataset, test_loader),
        }
        split_data = {k: v for k, v in split_data.items() if k in splits}
    else:
        dataset_class, path = DATASETS[dataset_name]
        full_dataset = dataset_class(path, data_filter=None)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        split_data = {"full": (full_dataset, full_loader)}

    _logger.info("Loading model...")
    model, device = load_model(model_name=model_name, checkpoint_path=model_checkpoint, compile=False)

    metrics_by_split = {}
    for split_name, (dataset, loader) in split_data.items():
        if len(dataset) == 0:
            _logger.warning(f"Split '{split_name}' is empty; skipping.")
            continue

        _logger.info(f"Evaluating split '{split_name}' ({len(dataset)} samples)...")
        metrics = evaluate_split(model, loader, device)
        metrics_by_split[split_name] = metrics
        _logger.info(f"[{split_name}] MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")

        recons_img, masks_img = plot_split_reconstructions(
            model=model,
            dataset=dataset,
            device=device,
            num_samples=num_samples,
            seed=seed,
        )
        recons_path = output_dir / f"{prefix}_{split_name}_reconstructions.png"
        recons_img.save(recons_path)
        _logger.info(f"Saved reconstructions to {recons_path}")

        if masks_img is not None:
            masks_path = output_dir / f"{prefix}_{split_name}_masks.png"
            masks_img.save(masks_path)
            _logger.info(f"Saved masks to {masks_path}")

    save_metrics_csv(output_dir=output_dir, prefix=prefix, metrics_by_split=metrics_by_split)
    _logger.info("Evaluation complete!")


def build_parser():
    p = ArgumentParser(description="Evaluate model reconstructions and plot reconstructions/masks")

    p.add_argument("--dataset_name", type=str, required=True,
        choices=list(DATASETS.keys()),
        help="Name of the dataset to use")
    p.add_argument("--filter_expr", type=str, default=None,
        help="Optional filter expression to partition train/test splits")
    p.add_argument("--model_name", type=str, required=True,
        choices=list(MODEL_TYPES.keys()),
        help="Type of model")
    p.add_argument("--model_checkpoint", type=Path, required=True,
        help="Path to model checkpoint")
    p.add_argument("--output_dir", type=Path, default=Path("plots/reconstructions"),
        help="Directory to save output files")
    p.add_argument("--batch_size", type=int, default=64,
        help="Batch size for evaluation")
    p.add_argument("--num_samples", type=int, default=8,
        help="Number of samples to plot per split")
    p.add_argument("--seed", type=int, default=42,
        help="Random seed for sampling plotted examples")
    p.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"],
        help="Which splits to evaluate when --filter_expr is provided")
    return p


parser = build_parser()


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
