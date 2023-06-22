import csv
import logging
from functools import partial
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Dinov2Model, AutoImageProcessor
from tqdm import tqdm

from scripts.common import DATASETS, MODEL_TYPES, load_model, load_dataset


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def load_dino(model_id, device, compile=True):
    model = Dinov2Model.from_pretrained(model_id).to(device)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    if compile:
        model = torch.compile(model)
    return model, processor


def get_image_features(img, model, processor, device):
    if img.shape[1] == 1:
        img = img.expand(-1, 3, -1, -1)
    inputs = processor(images=img, return_tensors="pt", do_rescale=False).to(device)
    features = model(**inputs)[0]
    return F.normalize(features, p=2, dim=-1)  # Normalize for cosine similarity


def compute_patchwise_similarity(emb1, emb2):
    B, Np, _ = emb1.shape
    cutoff_idx = int(np.ceil(Np * 0.05))
    emb1 = emb1.flatten(0, 1)
    emb2 = emb2.flatten(0, 1)
    similarities = torch.vmap(torch.dot)(emb1, emb2)  # Embeddings are already normalized
    similarities = similarities.unflatten(0, (B, Np))
    # return torch.min(similarities, dim=1)[0]
    similarities = torch.sort(similarities, dim=1)[0]
    return similarities[:, :cutoff_idx].mean(dim=1)


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    vision_model: str = 'facebook/dinov2-base',
    batch_size: int = 64,
):
    _logger.info(f"Loading {dataset_name} datasets...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size
    )
    _logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    _logger.info(f"Loading {model_name} model...")
    model, device = load_model(model_name, model_checkpoint)

    _logger.info(f"Loading {vision_model}...")
    vlm, processor = load_dino(vision_model, device)

    _logger.info("Setup processing...")
    _get_image_features = partial(
        get_image_features, model=vlm, processor=processor, device=device
    )
    def compare_images(recons, target):
        emb1 = _get_image_features(recons)
        emb2 = _get_image_features(target)
        similarity = compute_patchwise_similarity(emb1, emb2)
        return similarity.cpu().numpy()

    def compute_sims(loader):
        sims = []
        for x, y in tqdm(loader):
            r = model.reconstruction(x.to(device)).clip(0, 1).cpu()
            s = compare_images(r, y)
            sims.append(s)
        return np.concatenate(sims)

    _logger.info("Compute similarities...")
    train_similarities = compute_sims(train_loader)
    test_similarities = compute_sims(test_loader)

    print("=== CLIP Similarity Results ===")
    print(f"Train — mean: {train_similarities.mean():.4f}, std: {train_similarities.std():.4f}, "
                 f"min: {train_similarities.min():.4f}, max: {train_similarities.max():.4f}")
    print(f"Test  — mean: {test_similarities.mean():.4f}, std: {test_similarities.std():.4f}, "
                 f"min: {test_similarities.min():.4f}, max: {test_similarities.max():.4f}")

    _logger.info("Saving results...")
    output_path = model_checkpoint.parent / "clip_similarities.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "mean", "std", "min", "max"])
        for name, sims in [("train", train_similarities), ("test", test_similarities)]:
            writer.writerow([name, f"{sims.mean():.4f}", f"{sims.std():.4f}",
                             f"{sims.min():.4f}", f"{sims.max():.4f}"])

    _logger.info("Done!")

if __name__ == '__main__':
    parser = ArgumentParser(description="Latent factor prediction with linear regression")

    parser.add_argument("--dataset_name", type=str, required=True,
        choices=list(DATASETS.keys()),
        help="Name of the dataset to use")
    parser.add_argument("--filter_expr", type=str, required=True,
        help="Filter expression to determine train/test split")
    parser.add_argument("--model_name", type=str, required=True,
        choices=list(MODEL_TYPES.keys()),
        help="Type of model")
    parser.add_argument("--model_checkpoint", type=Path, required=True,
        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size for embedding computation")
    parser.add_argument("--vision_model", type=str, default='facebook/dinov2-base',
        help="Vision model used to analyse visual perception.")

    args = parser.parse_args()
    main(**vars(args))
