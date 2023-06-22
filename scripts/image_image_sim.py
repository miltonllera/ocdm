import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_model,
    load_dataset,
    load_vlm,
    tensor_to_pil,
    get_shape_name
)


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def evaluate_split(dataset, loader, model, vlm, processor, device, query_factor):
    factor_idx = list(dataset.factors).index(query_factor) if query_factor is not None else None
    shape_idx = list(dataset.factors).index('shape')

    details = []
    sample_offset = 0

    for x, _ in tqdm(loader, desc="Evaluating"):
        recons = model.reconstruction(x.to(device)).cpu()
        batch_size = x.shape[0]

        orig_imgs = [tensor_to_pil(x[i]) for i in range(batch_size)]
        recon_imgs = [tensor_to_pil(recons[i]) for i in range(batch_size)]

        combined_inputs = processor(images=orig_imgs + recon_imgs, return_tensors="pt").to(device)
        combined_feats = vlm.get_image_features(**combined_inputs)
        orig_feats, recon_feats = combined_feats.chunk(2, dim=0)

        orig_feats = orig_feats / orig_feats.norm(dim=-1, keepdim=True)
        recon_feats = recon_feats / recon_feats.norm(dim=-1, keepdim=True)
        sims = (orig_feats * recon_feats).sum(dim=-1)

        for i in range(batch_size):
            factor_vals = dataset.factor_values[sample_offset + i]
            shape_val = factor_vals[shape_idx]
            factor_val = factor_vals[factor_idx] if factor_idx is not None else None
            name = get_shape_name(dataset, shape_val)
            details.append({
                'sample_idx': sample_offset + i,
                'shape': name,
                'factor_value': factor_val,
                'similarity': sims[i].item(),
            })

        sample_offset += batch_size

    return details


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    query_factor: str,
    vlm_model: str,
    batch_size: int = 128,
):
    _logger.info(f"Loading {dataset_name} datasets...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size, num_workers=32,
    )
    _logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    _logger.info(f"Loading {model_name} model...")
    model, device = load_model(model_name, model_checkpoint)

    _logger.info(f"Loading VLM {vlm_model}...")
    vlm, processor = load_vlm(vlm_model, device)

    _logger.info("Evaluating train split...")
    train_details = evaluate_split(train_dataset, train_loader, model, vlm, processor, device, query_factor)

    _logger.info("Evaluating test split...")
    test_details = evaluate_split(test_dataset, test_loader, model, vlm, processor, device, query_factor)

    train_mean = np.mean([r['similarity'] for r in train_details])
    test_mean = np.mean([r['similarity'] for r in test_details])

    print("=== Image-Image Similarity Results ===")
    print(f"Train — mean similarity: {train_mean:.4f}")
    print(f"Test  — mean similarity: {test_mean:.4f}")

    output_path = model_checkpoint.parent / "image_image_sim.csv"
    _logger.info(f"Saving to {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_idx", "shape", "factor_value", "similarity"])
        for row in train_details:
            writer.writerow(["train", row['sample_idx'], row['shape'], row['factor_value'], row['similarity']])
        for row in test_details:
            writer.writerow(["test", row['sample_idx'], row['shape'], row['factor_value'], row['similarity']])

    _logger.info("Done!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Image-image cosine similarity between originals and reconstructions")

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
    parser.add_argument("--query_factor", type=str, required=False, default=None,
        help="Factor to record in output CSV (e.g., angle, orientation)")
    parser.add_argument("--vlm_model", type=str, default="google/siglip2-so400m-patch14-384",
        help="HuggingFace model ID for VLM image encoder")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Batch size for reconstruction")

    args = parser.parse_args()
    main(**vars(args))
