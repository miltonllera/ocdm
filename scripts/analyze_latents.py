import csv
import logging
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_or_compute_embeddings,
    load_model,
    load_dataset
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def bootstrap_score(score_fn, x, z, n_bootstrap=10, sample_size=10000):
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=min(sample_size, len(x)), replace=False)
        scores.append(score_fn(x[idx], z[idx], n_neighbors=15))
    return np.mean(scores), np.var(scores)


def run_pca_analysis(train_embeddings, output_dir, prefix, test_embeddings):
    _logger.info("Running PCA analysis...")
    if test_embeddings is not None:
        embeddings = np.concatenate([train_embeddings, test_embeddings])
    else:
        embeddings = train_embeddings

    pca = PCA().fit(embeddings)
    train_proj = pca.transform(train_embeddings)
    test_proj = pca.transform(test_embeddings) if test_embeddings is not None else None

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Bar(y=pca.explained_variance_ratio_, name="Explained variance ratio"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(y=np.cumsum(pca.explained_variance_ratio_), name="Cumulative explained variance"),
        row=1, col=1
    )
    fig.update_layout(
        template='plotly_white',
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Ratio"
    )
    fig.write_html(output_dir / f"{prefix}_pca_explained_variance.html")
    fig.write_image(output_dir / f"{prefix}_pca_explained_variance.png")

    return pca, train_proj, test_proj


def run_umap_analysis(
    train_embeddings,
    output_dir,
    prefix,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    test_embeddings=None
):
    _logger.info("Running UMAP analysis...")
    _logger.info(f"UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )

    if test_embeddings is not None:
        embeddings = np.concatenate([train_embeddings, test_embeddings])
    else:
        embeddings = train_embeddings

    umap_proj = reducer.fit_transform(embeddings)

    _logger.info("Computing trustworthiness and continuity...")
    T, T_var = bootstrap_score(partial(trustworthiness, metric=metric), embeddings, umap_proj)
    C, C_var = bootstrap_score(partial(trustworthiness, metric=metric), umap_proj, embeddings)

    _logger.info(f"Trustworthiness: {T:.4f} +/- {np.sqrt(T_var):.4f}")
    _logger.info(f"Continuity: {C:.4f} +/- {np.sqrt(C_var):.4f}")

    fig = px.bar(
        x=["Trustworthiness", "Continuity"],
        y=[T, C],
        error_y=[np.sqrt(T_var), np.sqrt(C_var)],
        range_y=[0.0, 1.0],
        title="UMAP Quality Metrics"
    )
    fig.update_layout(template='plotly_white', yaxis_title="Score")
    fig.write_html(output_dir / f"{prefix}_umap_metrics.html")
    fig.write_image(output_dir / f"{prefix}_umap_metrics.png")

    metrics = {
        'trustworthiness_mean': T,
        'trustworthiness_var': T_var,
        'continuity_mean': C,
        'continuity_var': C_var,
    }

    if test_embeddings is not None:
        umap_proj, test_umap_proj = (
            umap_proj[:len(train_embeddings)],  # type: ignore
            umap_proj[len(train_embeddings):]  # type: ignore
        )
    else:
        test_umap_proj = None

    return umap_proj, test_umap_proj, metrics


def _create_continuous_scatter(train_emb, test_emb, train_color, test_color):
    fig = go.Figure()
    x, y, z = train_emb.T[:3]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3, color=train_color, opacity=0.3, coloraxis='coloraxis'),
        name='Train',
        legendgroup='train',
        showlegend=True
    ))

    if test_emb is not None:
        test_x, test_y, test_z = test_emb.T[:3]

        fig.add_trace(go.Scatter3d(
            x=test_x, y=test_y, z=test_z,
            mode='markers',
            marker=dict(size=3, color=test_color, opacity=0.3,
                        symbol='diamond', coloraxis='coloraxis'),
            name='Test',
            legendgroup='test',
            showlegend=True
        ))

    return fig


def _create_per_class_scatter(train_emb, test_emb, train_color, test_color, color_map):
    fig = go.Figure()

    def plot_embeddings(embeddings, color, split):
        x, y, z = embeddings.T[:3]
        color = color.astype(int)
        classes = np.unique(color)
        for c in classes:
            fig.add_trace(
                go.Scatter3d(
                    name=f"{color_map(c)}" + ("_train" if split == "train" else ""),
                    mode='markers',
                    x=x[color==c],
                    y=y[color==c],
                    z=z[color==c],
                    marker=dict(
                        color=c,
                        symbol='circle' if split == 'train' else 'diamond',
                        size=5,
                        showscale=False,
                        opacity=0.3,
                    ),
                )
            )

    plot_embeddings(train_emb, train_color, 'train')
    if test_emb is not None:
        plot_embeddings(test_emb, test_color, 'test')

    return fig


def create_interactive_scatter(
    projection_type,
    projection,
    factor_values,
    factor_names,
    dataset,
    output_dir,
    prefix,
    sample_size=10000,
    test_projection=None,
    test_factor_values=None
):
    _logger.info(f"Creating interactive {projection_type.upper()} scatter plots...")

    if len(projection) > sample_size:
        idx = np.random.choice(len(projection), size=sample_size, replace=False)
        proj_sample = projection[idx]
        factors_sample = factor_values[idx]
    else:
        proj_sample = projection
        factors_sample = factor_values

    test_proj_sample = None
    test_factors_sample = None
    if test_projection is not None and test_factor_values is not None:
        test_proj_sample = len(test_projection) * sample_size // len(projection)
        if len(test_projection) > test_proj_sample:
            test_idx = np.random.choice(len(test_projection), size=test_proj_sample, replace=False)
            test_proj_sample = test_projection[test_idx]
            test_factors_sample = test_factor_values[test_idx]
        else:
            test_proj_sample = test_projection
            test_factors_sample = test_factor_values

    for i, factor_name in enumerate(factor_names):
        color = factors_sample[:, i]

        if factor_name == 'shape':
            fig = _create_per_class_scatter(
                proj_sample,
                test_proj_sample,
                color,
                test_factors_sample[:, i] if test_factors_sample is not None else None,
                dataset.map_shapes
            )
        else:
            fig = _create_continuous_scatter(
                proj_sample,
                test_proj_sample,
                color,
                test_factors_sample[:, i] if test_factors_sample is not None else None,
            )

        fig.update_layout(
            template='plotly_white',
            height=800,
            width=800,
            title=f"{projection_type.upper()} colored by {factor_name}",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        fig.write_html(output_dir / f"{prefix}_{projection_type.lower()}_{factor_name}.html")


def save_results_csv(output_dir, prefix, pca, umap_metrics, factor_names):
    csv_path = output_dir / f"{prefix}_results.csv"

    with open(csv_path, 'w', newline='') as f:
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
            umap_metrics['trustworthiness_mean'],
            umap_metrics['trustworthiness_var'],
            np.sqrt(umap_metrics['trustworthiness_var'])
        ])
        writer.writerow([
            "Continuity",
            umap_metrics['continuity_mean'],
            umap_metrics['continuity_var'],
            np.sqrt(umap_metrics['continuity_var'])
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
    umap_metric: str = 'euclidean',
    include_test: bool = False,
):
    output_dir = output_dir / model_checkpoint.parts[-2]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset_name}"
    embeddings_path = model_checkpoint.parent / "embeddings.npz"

    _logger.info("Loading dataset...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size
    )
    factor_values = train_dataset.factor_values
    factor_names = type(train_dataset).factors

    _logger.info("Loading model...")
    model, device = load_model(model_name, model_checkpoint)

    embeddings = load_or_compute_embeddings(
        embeddings_path, model, train_loader, device, force_recompute, "train"
    )
    _logger.info(f"Embeddings shape: {embeddings.shape}")

    test_embeddings = None
    test_factor_values = None
    if include_test:
        test_embeddings = load_or_compute_embeddings(
            embeddings_path, model, test_loader, device, force_recompute, "test"
        )
        test_factor_values = test_dataset.factor_values
        _logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    pca, train_pca_proj, test_pca_proj = run_pca_analysis(
        embeddings, output_dir, prefix, test_embeddings=test_embeddings
    )

    umap_proj, test_umap_proj, umap_metrics = run_umap_analysis(
        embeddings, output_dir, prefix,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        test_embeddings=test_embeddings
    )

    create_interactive_scatter(
        "PCA",
        train_pca_proj, factor_values, factor_names, train_dataset, output_dir, prefix,
        test_projection=test_pca_proj,
        test_factor_values=test_factor_values
    )
    create_interactive_scatter(
        "UMAP",
        umap_proj, factor_values, factor_names, train_dataset, output_dir, prefix,
        test_projection=test_umap_proj,
        test_factor_values=test_factor_values
    )

    save_results_csv(output_dir, prefix, pca, umap_metrics, factor_names)

    _logger.info("Analysis complete!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Latent space analysis with PCA and UMAP")

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
    parser.add_argument("--output_dir", type=Path, default=Path("plots/latent_analysis"),
        help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Batch size for embedding computation")
    parser.add_argument("--force_recompute", action="store_true",
        help="Force recomputation of embeddings even if cached file exists")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
        help="Number of neighbors for UMAP (default: 15)")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
        help="Minimum distance for UMAP (default: 0.1)")
    parser.add_argument("--umap_metric", type=str, default='euclidean',
        help="Distance metric for UMAP (default: euclidean)")
    parser.add_argument("--include_test", action="store_true",
        help="Include test embeddings in UMAP projection (plotted with different marker)")

    args = parser.parse_args()
    main(**vars(args))
