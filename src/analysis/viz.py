from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def strip(ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def create_latent_scatter(
    train_proj: np.ndarray,
    train_color: np.ndarray,
    test_proj: np.ndarray | None = None,
    test_color: np.ndarray | None = None,
    factor_name: str = "Factor",
    is_categorical: bool = False,
    label_map=None,
    is_3d: bool = True,
    marker_size: int = 3,
    opacity: float = 0.4,
    colorscale: str = "Viridis",
    title: str = "Latent Space Projection",
):
    fig = go.Figure()

    if is_categorical:
        train_labels = train_color.astype(int)
        unique_classes = np.unique(train_labels)

        for c in unique_classes:
            c_name = label_map(c) if label_map is not None else f"Class {c}"
            mask = train_labels == c
            if is_3d:
                x, y, z = train_proj[mask, 0], train_proj[mask, 1], train_proj[mask, 2]
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="markers",
                    name=f"{c_name} (Train)",
                    marker=dict(size=marker_size, opacity=opacity, symbol="circle"),
                ))
            else:
                x, y = train_proj[mask, 0], train_proj[mask, 1]
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="markers",
                    name=f"{c_name} (Train)",
                    marker=dict(size=marker_size, opacity=opacity, symbol="circle"),
                ))

        if test_proj is not None and test_color is not None:
            test_labels = test_color.astype(int)
            test_unique = np.unique(test_labels)
            for c in test_unique:
                c_name = label_map(c) if label_map is not None else f"Class {c}"
                mask = test_labels == c
                if is_3d:
                    x, y, z = test_proj[mask, 0], test_proj[mask, 1], test_proj[mask, 2]
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode="markers",
                        name=f"{c_name} (Test)",
                        marker=dict(size=marker_size + 2, opacity=opacity + 0.2, symbol="diamond"),
                    ))
                else:
                    x, y = test_proj[mask, 0], test_proj[mask, 1]
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="markers",
                        name=f"{c_name} (Test)",
                        marker=dict(size=marker_size + 2, opacity=opacity + 0.2, symbol="diamond"),
                    ))
    else:
        if is_3d:
            x, y, z = train_proj[:, 0], train_proj[:, 1], train_proj[:, 2]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="Train",
                marker=dict(
                    size=marker_size,
                    color=train_color,
                    colorscale=colorscale,
                    colorbar=dict(title=factor_name),
                    opacity=opacity,
                    symbol="circle",
                ),
            ))
            if test_proj is not None and test_color is not None:
                tx, ty, tz = test_proj[:, 0], test_proj[:, 1], test_proj[:, 2]
                fig.add_trace(go.Scatter3d(
                    x=tx, y=ty, z=tz,
                    mode="markers",
                    name="Test",
                    marker=dict(
                        size=marker_size + 2,
                        color=test_color,
                        colorscale=colorscale,
                        opacity=min(1.0, opacity + 0.3),
                        symbol="diamond",
                    ),
                ))
        else:
            x, y = train_proj[:, 0], train_proj[:, 1]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                name="Train",
                marker=dict(
                    size=marker_size,
                    color=train_color,
                    colorscale=colorscale,
                    colorbar=dict(title=factor_name),
                    opacity=opacity,
                    symbol="circle",
                ),
            ))
            if test_proj is not None and test_color is not None:
                tx, ty = test_proj[:, 0], test_proj[:, 1]
                fig.add_trace(go.Scatter(
                    x=tx, y=ty,
                    mode="markers",
                    name="Test",
                    marker=dict(
                        size=marker_size + 2,
                        color=test_color,
                        colorscale=colorscale,
                        opacity=min(1.0, opacity + 0.3),
                        symbol="diamond",
                    ),
                ))

    layout_args = dict(
        template="plotly_white",
        title=title,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40),
        height=750,
    )
    if is_3d:
        layout_args["scene"] = dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        )
    else:
        layout_args["xaxis_title"] = "Dim 1"
        layout_args["yaxis_title"] = "Dim 2"

    fig.update_layout(**layout_args)
    return fig


def create_pca_variance_plot(explained_variance_ratio: np.ndarray):
    fig = make_subplots(rows=1, cols=1)
    cum_var = np.cumsum(explained_variance_ratio)
    dims = list(range(1, len(explained_variance_ratio) + 1))

    fig.add_trace(
        go.Bar(x=dims, y=explained_variance_ratio, name="Individual Explained Var"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dims, y=cum_var, mode="lines+markers", name="Cumulative Explained Var"),
        row=1, col=1
    )
    fig.update_layout(
        template="plotly_white",
        title="PCA Explained Variance by Component",
        xaxis_title="Principal Component",
        yaxis_title="Variance Ratio",
        yaxis=dict(range=[0, 1.05]),
        height=400,
    )
    return fig


def create_umap_metrics_plot(metrics: dict):
    t_mean = metrics["trustworthiness_mean"]
    t_std = np.sqrt(metrics["trustworthiness_var"])
    c_mean = metrics["continuity_mean"]
    c_std = np.sqrt(metrics["continuity_var"])

    fig = go.Figure(data=[
        go.Bar(
            name="Metrics",
            x=["Trustworthiness", "Continuity"],
            y=[t_mean, c_mean],
            error_y=dict(type="data", array=[t_std, c_std], visible=True),
            marker_color=["#4C78A8", "#F58518"],
        )
    ])
    fig.update_layout(
        template="plotly_white",
        title="UMAP Embedding Quality Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.0]),
        height=350,
    )
    return fig


def create_dataset_grid_image(dataset, num_samples: int = 16, seed: int = 42):
    rng = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    indices = torch.randperm(total_len, generator=rng)[:num_samples].tolist()

    images = []
    for idx in indices:
        sample = dataset[idx]
        img = sample[0] if isinstance(sample, tuple) else sample
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        images.append(img.float())

    images_tensor = torch.stack(images, dim=0)
    nrow = int(len(images) ** 0.5)
    if nrow * nrow < len(images):
        nrow += 1

    grid = vutils.make_grid(images_tensor, nrow=nrow, padding=2, normalize=True)
    grid_np = (grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(grid_np)


def create_reconstruction_grid_image(model, dataset, num_samples: int = 8, seed: int = 42, device: str = "cpu"):
    rng = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    indices = torch.randperm(total_len, generator=rng)[:num_samples].tolist()

    images = []
    recons = []
    model = model.to(torch.device(device))
    model.eval()

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            orig = sample[0] if isinstance(sample, tuple) else sample
            x = orig.unsqueeze(0).to(torch.device(device))
            recon = model(x)[0]
            if isinstance(recon, (tuple, list)):
                recon = recon[0]
            recon = recon.squeeze(0).cpu().clamp(0, 1)

            if orig.shape[0] == 1:
                orig = orig.repeat(3, 1, 1)
            if recon.shape[0] == 1:
                recon = recon.repeat(3, 1, 1)

            images.append(orig.float())
            recons.append(recon.float())

    # Interleave original and reconstruction
    paired = []
    for o, r in zip(images, recons):
        paired.append(o)
        paired.append(r)

    paired_tensor = torch.stack(paired, dim=0)
    grid = vutils.make_grid(paired_tensor, nrow=4, padding=4, normalize=True)
    grid_np = (grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(grid_np)
