import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModel

from scripts.analyze_latents import save_results_csv
from scripts.common import (
    DATASETS,
    load_dataset,
    load_or_compute_embeddings,
)
from src.analysis.latents import (
    run_pca_reduction,
    run_umap_reduction,
    subsample_points,
)
from src.analysis.viz import (
    create_latent_scatter,
    create_pca_variance_plot,
    create_umap_metrics_plot,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class HFVisionEmbedder:
    """Wrapper around a HuggingFace vision model exposing `.embed(x)` for `compute_embeddings`."""

    def __init__(self, model_id: str, device: torch.device):
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _try_write_image(fig, path: Path) -> None:
    try:
        fig.write_image(str(path))
    except Exception as exc:
        _logger.warning(f"Could not export static image {path.name}: {exc}")


def _save_pca_csv(output_dir: Path, prefix: str, pca, factor_names: list[str]) -> None:
    csv_path = output_dir / f"{prefix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PCA Results"])
        writer.writerow(["Component", "Explained Variance Ratio", "Cumulative Variance"])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumsum)):
            writer.writerow([i + 1, var, cum])
        writer.writerow([])
        writer.writerow(["Factor Names"])
        writer.writerow(factor_names)
    _logger.info(f"Results saved to {csv_path}")


def main(
    dataset_name: str,
    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    dataset_path: str | None = None,
    filter_expr: str | None = None,
    output_dir: Path = Path("plots/hf_latent_analysis"),
    batch_size: int = 64,
    num_workers: int = 4,
    force_recompute: bool = False,
    sample_size: int = 10000,
    max_samples: int | None = None,
    run_umap: bool = False,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_subsample_ratio: float = 1.0,
):
    if not (0.0 < umap_subsample_ratio <= 1.0):
        raise ValueError(f"umap_subsample_ratio must be in (0, 1], got {umap_subsample_ratio}")

    model_slug = model_id.replace("/", "__")
    dataset_tag = f"{dataset_name}_n{max_samples}" if max_samples is not None else dataset_name
    run_dir = output_dir / model_slug / dataset_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_slug}_{dataset_tag}"
    embeddings_path = run_dir / "embeddings.npz"

    _logger.info(f"Loading dataset '{dataset_name}'...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name=dataset_name,
        filter_expr=filter_expr,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_path=dataset_path,
    )
    factor_values = train_dataset.factor_values
    factor_names = list(type(train_dataset).factors)
    if max_samples is not None and len(train_dataset) > max_samples:
        rng = np.random.default_rng(42)
        sub_idx = np.sort(rng.choice(len(train_dataset), size=max_samples, replace=False))
        factor_values = factor_values[sub_idx]
        train_loader = DataLoader(
            Subset(train_dataset, sub_idx),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        _logger.info(f"Subsampled to {len(sub_idx)} / {len(train_dataset)} samples with factors: {factor_names}")
    else:
        _logger.info(f"Loaded {len(train_dataset)} samples with factors: {factor_names}")

    device = _get_device()
    _logger.info(f"Loading HuggingFace model '{model_id}' on {device}...")
    embedder = HFVisionEmbedder(model_id=model_id, device=device)

    embeddings = load_or_compute_embeddings(
        embeddings_path=embeddings_path,
        model=embedder,
        loader=train_loader,
        device=device,
        force_recompute=force_recompute,
        split_name="train",
    )
    _logger.info(f"Embeddings shape: {embeddings.shape}")

    test_embeddings = None
    test_factor_values = None
    if test_dataset is not None and test_loader is not None:
        test_embeddings = load_or_compute_embeddings(
            embeddings_path=embeddings_path,
            model=embedder,
            loader=test_loader,
            device=device,
            force_recompute=force_recompute,
            split_name="test",
        )
        test_factor_values = test_dataset.factor_values
        _logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    _logger.info("Running PCA analysis...")
    pca_res = run_pca_reduction(
        train_embeddings=embeddings, test_embeddings=test_embeddings, n_components=3
    )
    evr = pca_res["explained_variance_ratio"]
    _logger.info(
        f"PCA explained variance (PC1-3): {evr[:3].round(4).tolist()} "
        f"(cumulative: {evr[:3].sum():.4f})"
    )
    train_proj = pca_res["train_proj"]
    for i, fname in enumerate(factor_names):
        vals = factor_values[:, i]
        if np.std(vals) < 1e-8:
            _logger.info(f"  Factor '{fname}': constant (std=0)")
            continue
        if fname in ("shape", "object"):
            # Variance explained (eta^2) of each PC by categorical factor
            classes = np.unique(vals)
            eta2 = []
            for pc in range(3):
                pc_vals = train_proj[:, pc]
                ss_between = sum(
                    np.sum(vals == c) * (np.mean(pc_vals[vals == c]) - np.mean(pc_vals)) ** 2
                    for c in classes
                )
                ss_total = np.sum((pc_vals - np.mean(pc_vals)) ** 2)
                eta2.append(float(ss_between / ss_total) if ss_total > 0 else 0.0)
            _logger.info(f"  Factor '{fname}' (categorical) PC1-3 eta^2: {[round(x, 4) for x in eta2]}")
        else:
            corrs = [float(np.corrcoef(train_proj[:, pc], vals)[0, 1]) for pc in range(3)]
            _logger.info(f"  Factor '{fname}' PC1-3 Pearson r: {[round(x, 4) for x in corrs]}")
    pca_fig = create_pca_variance_plot(evr)
    pca_fig.write_html(str(run_dir / f"{prefix}_pca_explained_variance.html"))
    _try_write_image(pca_fig, run_dir / f"{prefix}_pca_explained_variance.png")

    train_pca_sub, train_pca_factors_sub, _ = subsample_points(
        pca_res["train_proj"], factor_values, sample_size
    )
    test_pca_sub, test_pca_factors_sub = None, None
    if test_embeddings is not None and test_factor_values is not None:
        test_size = max(100, int(len(test_embeddings) * (sample_size / len(embeddings))))
        test_pca_sub, test_pca_factors_sub, _ = subsample_points(
            pca_res["test_proj"], test_factor_values, test_size
        )

    umap_res = None
    train_umap_sub, train_umap_factors_sub = None, None
    test_umap_sub, test_umap_factors_sub = None, None
    if run_umap:
        if umap_subsample_ratio < 1.0:
            umap_train_size = max(1, int(len(embeddings) * umap_subsample_ratio))
            umap_train_emb, umap_train_factors, _ = subsample_points(
                embeddings, factor_values, umap_train_size
            )
            if test_embeddings is not None and test_factor_values is not None:
                umap_test_size = max(1, int(len(test_embeddings) * umap_subsample_ratio))
                umap_test_emb, umap_test_factors, _ = subsample_points(
                    test_embeddings, test_factor_values, umap_test_size
                )
            else:
                umap_test_emb, umap_test_factors = None, None
        else:
            umap_train_emb, umap_train_factors = embeddings, factor_values
            umap_test_emb, umap_test_factors = test_embeddings, test_factor_values

        umap_res = run_umap_reduction(
            train_embeddings=umap_train_emb,
            test_embeddings=umap_test_emb,
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            compute_metrics=True,
        )
        umap_fig = create_umap_metrics_plot(umap_res["metrics"])
        umap_fig.write_html(str(run_dir / f"{prefix}_umap_metrics.html"))
        _try_write_image(umap_fig, run_dir / f"{prefix}_umap_metrics.png")

        train_umap_sub, train_umap_factors_sub, _ = subsample_points(
            umap_res["train_proj"], umap_train_factors, sample_size
        )
        if umap_test_emb is not None and umap_test_factors is not None:
            umap_plot_test_size = max(100, int(len(umap_test_emb) * (sample_size / len(umap_train_emb))))
            test_umap_sub, test_umap_factors_sub, _ = subsample_points(
                umap_res["test_proj"], umap_test_factors, umap_plot_test_size
            )

    label_map = train_dataset.map_shapes if hasattr(train_dataset, "map_shapes") else None
    for i, factor_name in enumerate(factor_names):
        is_categorical = factor_name in ("shape", "object")

        pca_scatter = create_latent_scatter(
            train_proj=train_pca_sub,
            train_color=train_pca_factors_sub[:, i],
            test_proj=test_pca_sub,
            test_color=test_pca_factors_sub[:, i] if test_pca_factors_sub is not None else None,
            factor_name=factor_name,
            is_categorical=is_categorical,
            label_map=label_map,
            is_3d=True,
            title=f"{model_id} PCA colored by {factor_name}",
        )
        pca_scatter.write_html(str(run_dir / f"{prefix}_pca_{factor_name}.html"))

        if run_umap and train_umap_sub is not None:
            umap_scatter = create_latent_scatter(
                train_proj=train_umap_sub,
                train_color=train_umap_factors_sub[:, i],
                test_proj=test_umap_sub,
                test_color=test_umap_factors_sub[:, i] if test_umap_factors_sub is not None else None,
                factor_name=factor_name,
                is_categorical=is_categorical,
                label_map=label_map,
                is_3d=True,
                title=f"{model_id} UMAP colored by {factor_name}",
            )
            umap_scatter.write_html(str(run_dir / f"{prefix}_umap_{factor_name}.html"))

    if umap_res is not None:
        save_results_csv(
            output_dir=run_dir,
            prefix=prefix,
            pca=pca_res["pca"],
            umap_metrics=umap_res["metrics"],
            factor_names=factor_names,
        )
    else:
        _save_pca_csv(
            output_dir=run_dir,
            prefix=prefix,
            pca=pca_res["pca"],
            factor_names=factor_names,
        )
    _logger.info(f"Analysis complete! Outputs in {run_dir}")


def build_parser():
    p = ArgumentParser(description="Latent space analysis for HuggingFace vision models")
    p.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Name of the dataset to use",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional override path for the dataset",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="HuggingFace vision model ID",
    )
    p.add_argument(
        "--filter_expr",
        type=str,
        default=None,
        help="Optional filter expression for train/test split (omit to analyze full dataset)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plots/hf_latent_analysis"),
        help="Directory to save output files",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--sample_size", type=int, default=10000)
    p.add_argument("--max_samples", type=int, default=None, help="Optional max samples to embed from dataset")
    p.add_argument("--run_umap", action="store_true")
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="euclidean")
    p.add_argument("--umap_subsample_ratio", type=float, default=1.0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(**vars(args))
