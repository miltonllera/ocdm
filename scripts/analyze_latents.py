import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_or_compute_embeddings,
    load_model,
    load_dataset,
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


def save_results_csv(output_dir: Path, prefix: str, pca, umap_metrics: dict, factor_names: list[str]):
    csv_path = output_dir / f"{prefix}_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["PCA Results"])
        writer.writerow(["Component", "Explained Variance Ratio", "Cumulative Variance"])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumsum)):
            writer.writerow([i + 1, var, cum])

        writer.writerow([])

        writer.writerow(["UMAP Metrics"])
        writer.writerow(["Metric", "Mean", "Variance", "Std"])
        writer.writerow([
            "Trustworthiness",
            umap_metrics["trustworthiness_mean"],
            umap_metrics["trustworthiness_var"],
            np.sqrt(umap_metrics["trustworthiness_var"]),
        ])
        writer.writerow([
            "Continuity",
            umap_metrics["continuity_mean"],
            umap_metrics["continuity_var"],
            np.sqrt(umap_metrics["continuity_var"]),
        ])

        writer.writerow([])

        writer.writerow(["Factor Names"])
        writer.writerow(factor_names)

    _logger.info(f"Results saved to {csv_path}")


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    output_dir: Path,
    batch_size: int = 64,
    force_recompute: bool = False,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    include_test: bool = False,
    sample_size: int = 10000,
):
    output_dir = output_dir / model_checkpoint.parts[-2]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset_name}"
    embeddings_path = model_checkpoint.parent / "embeddings.npz"

    _logger.info("Loading dataset...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name=dataset_name,
        filter_expr=filter_expr,
        batch_size=batch_size,
    )
    factor_values = train_dataset.factor_values
    factor_names = type(train_dataset).factors

    _logger.info("Loading model...")
    model, device = load_model(model_name=model_name, checkpoint_path=model_checkpoint, compile=False)

    embeddings = load_or_compute_embeddings(
        embeddings_path=embeddings_path,
        model=model,
        loader=train_loader,
        device=device,
        force_recompute=force_recompute,
        split_name="train",
    )
    _logger.info(f"Embeddings shape: {embeddings.shape}")

    test_embeddings = None
    test_factor_values = None
    if include_test:
        test_embeddings = load_or_compute_embeddings(
            embeddings_path=embeddings_path,
            model=model,
            loader=test_loader,
            device=device,
            force_recompute=force_recompute,
            split_name="test",
        )
        test_factor_values = test_dataset.factor_values
        _logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    _logger.info("Running PCA analysis on full dataset...")
    pca_res = run_pca_reduction(train_embeddings=embeddings, test_embeddings=test_embeddings, n_components=3)
    pca_fig = create_pca_variance_plot(pca_res["explained_variance_ratio"])
    pca_fig.write_html(str(output_dir / f"{prefix}_pca_explained_variance.html"))
    pca_fig.write_image(str(output_dir / f"{prefix}_pca_explained_variance.png"))

    _logger.info("Running UMAP analysis on full dataset...")
    umap_res = run_umap_reduction(
        train_embeddings=embeddings,
        test_embeddings=test_embeddings,
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        compute_metrics=True,
    )
    umap_fig = create_umap_metrics_plot(umap_res["metrics"])
    umap_fig.write_html(str(output_dir / f"{prefix}_umap_metrics.html"))
    umap_fig.write_image(str(output_dir / f"{prefix}_umap_metrics.png"))

    # Subsample full projections for scatter plotting
    train_pca_sub, train_factors_sub, _ = subsample_points(pca_res["train_proj"], factor_values, sample_size)
    train_umap_sub, _, _ = subsample_points(umap_res["train_proj"], factor_values, sample_size)

    test_pca_sub = None
    test_umap_sub = None
    test_factors_sub = None
    if test_embeddings is not None and test_factor_values is not None:
        test_size = max(100, int(len(test_embeddings) * (sample_size / len(embeddings))))
        test_pca_sub, test_factors_sub, _ = subsample_points(pca_res["test_proj"], test_factor_values, test_size)
        test_umap_sub, _, _ = subsample_points(umap_res["test_proj"], test_factor_values, test_size)

    # Generate factor scatter plots
    for i, factor_name in enumerate(factor_names):
        is_categorical = factor_name == "shape"
        label_map = train_dataset.map_shapes if hasattr(train_dataset, "map_shapes") else None

        train_c = train_factors_sub[:, i]
        test_c = test_factors_sub[:, i] if test_factors_sub is not None else None

        pca_scatter = create_latent_scatter(
            train_proj=train_pca_sub,
            train_color=train_c,
            test_proj=test_pca_sub,
            test_color=test_c,
            factor_name=factor_name,
            is_categorical=is_categorical,
            label_map=label_map,
            is_3d=True,
            title=f"PCA colored by {factor_name}",
        )
        pca_scatter.write_html(str(output_dir / f"{prefix}_pca_{factor_name}.html"))

        umap_scatter = create_latent_scatter(
            train_proj=train_umap_sub,
            train_color=train_c,
            test_proj=test_umap_sub,
            test_color=test_c,
            factor_name=factor_name,
            is_categorical=is_categorical,
            label_map=label_map,
            is_3d=True,
            title=f"UMAP colored by {factor_name}",
        )
        umap_scatter.write_html(str(output_dir / f"{prefix}_umap_{factor_name}.html"))

    save_results_csv(
        output_dir=output_dir,
        prefix=prefix,
        pca=pca_res["pca"],
        umap_metrics=umap_res["metrics"],
        factor_names=factor_names,
    )
    _logger.info("Analysis complete!")


def build_parser():
    p = ArgumentParser(description="Latent space analysis with PCA and UMAP")

    p.add_argument("--dataset_name", type=str, required=True,
        choices=list(DATASETS.keys()),
        help="Name of the dataset to use")
    p.add_argument("--filter_expr", type=str, required=True,
        help="Filter expression to determine train/test split")
    p.add_argument("--model_name", type=str, required=True,
        choices=list(MODEL_TYPES.keys()),
        help="Type of model")
    p.add_argument("--model_checkpoint", type=Path, required=True,
        help="Path to model checkpoint")
    p.add_argument("--output_dir", type=Path, default=Path("plots/latent_analysis"),
        help="Directory to save output files")
    p.add_argument("--batch_size", type=int, default=64,
        help="Batch size for embedding computation")
    p.add_argument("--force_recompute", action="store_true",
        help="Force recomputation of embeddings even if cached file exists")
    p.add_argument("--umap_n_neighbors", type=int, default=15,
        help="Number of neighbors for UMAP (default: 15)")
    p.add_argument("--umap_min_dist", type=float, default=0.1,
        help="Minimum distance for UMAP (default: 0.1)")
    p.add_argument("--umap_metric", type=str, default="euclidean",
        help="Distance metric for UMAP (default: euclidean)")
    p.add_argument("--include_test", action="store_true",
        help="Include test embeddings in UMAP projection")
    p.add_argument("--sample_size", type=int, default=10000,
        help="Sample size for scatter plots")
    return p


parser = build_parser()


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
