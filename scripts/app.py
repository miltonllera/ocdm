import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import torch
import plotly.express as px

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_model,
    load_dataset,
    compute_embeddings,
)
from src.dataset.multi_shape import ColoredMultiShapeDataset
from src.analysis.latents import (
    find_checkpoints,
    subsample_points,
    load_cached_embeddings,
    run_reduction_in_process,
    cancel_active_reduction_process,
)
from src.analysis.viz import (
    create_latent_scatter,
    create_pca_variance_plot,
    create_umap_metrics_plot,
    create_dataset_grid_image,
    create_reconstruction_grid_image,
)

st.set_page_config(
    page_title="OCDM Representation Learning Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def infer_model_and_dataset(ckpt_path: Path):
    path_str = ckpt_path.as_posix()
    default_model = "vae"
    for m in MODEL_TYPES:
        if f"_{m}_" in path_str or f"/{m}/" in path_str or f"_{m}" in path_str:
            default_model = m
            break
    default_ds = "dsprites"
    for d in DATASETS:
        if d in path_str:
            default_ds = d
            break
    return default_model, default_ds


@st.cache_resource
def get_cached_model(model_name: str, checkpoint_path: str, device: str = "cpu"):
    model, _ = load_model(
        model_name=model_name, checkpoint_path=Path(checkpoint_path), compile=False
    )
    model = model.to(torch.device(device))
    return model


@st.cache_data
def get_cached_dataset(dataset_name: str, filter_expr: str, batch_size: int = 64):
    train_ds, test_ds, train_loader, test_loader = load_dataset(
        dataset_name=dataset_name,
        filter_expr=filter_expr,
        batch_size=batch_size,
        num_workers=0,
    )
    return train_ds, test_ds


def main():
    st.sidebar.title("🔬 OCDM Explorer")
    st.sidebar.markdown("Inspect datasets, model checkpoints, and latent manifolds.")

    mode = st.sidebar.radio(
        "Navigation Mode",
        ["Latent Space Analysis", "Dataset Inspection", "Reconstruction Quality"],
        index=0,
    )

    if mode == "Dataset Inspection":
        render_dataset_inspection()
    elif mode == "Latent Space Analysis":
        render_latent_analysis()
    elif mode == "Reconstruction Quality":
        render_reconstructions()


def render_dataset_inspection():
    st.header("🖼️ Dataset Inspection")
    st.markdown(
        "Examine raw dataset samples, configurations, and ground-truth generative factor distributions."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dataset Configuration")
        dataset_type = st.selectbox(
            "Dataset",
            [
                "dsprites",
                "pentominos",
                "shapes3d",
                "multi_pentominos",
                "multi_dsprites",
            ],
            index=0,
        )

        num_samples = st.slider(
            "Number of samples to visualize",
            min_value=4,
            max_value=64,
            value=16,
            step=4,
        )
        seed = st.number_input("Random Seed", value=42, step=1)

        filter_expr = ""
        if dataset_type in DATASETS:
            preset_filters = {
                "None (Full Dataset)": "",
                "dSprites Heart Pos Held-out: (shape == 3) & (posX < 0.5)": "( shape == 3 ) & ( posX < 0.50 )",
                "dSprites Heart Rot Held-out: (shape == 3) & (orientation < np.pi)": "( shape == 3 ) & ( orientation < np.pi )",
                "Pentominos Rotation: np.isin(shape, [1, 3, 5, 8]) & (angle >= 180)": "np.isin( shape , [1, 3, 5, 8] ) & ( angle >= 180 )",
            }
            filter_choice = st.selectbox(
                "Quick Filter Presets", list(preset_filters.keys())
            )
            default_expr = preset_filters[filter_choice]
            filter_expr = st.text_input(
                "Filter Expression (leaves blank for full)", value=default_expr
            )

        multi_n_shapes = 2
        multi_num_colors = 10
        if "multi_" in dataset_type:
            multi_n_shapes = st.slider(
                "Number of Shapes", min_value=1, max_value=5, value=2
            )
            multi_num_colors = st.slider(
                "Number of Colors", min_value=2, max_value=15, value=10
            )

    with col2:
        st.subheader("Sample Grid")
        # Load dataset
        if dataset_type in ("multi_pentominos", "multi_dsprites"):
            base_name = "pentominos" if "pentominos" in dataset_type else "dsprites"
            base_cls, base_path = DATASETS[base_name]
            base_ds = base_cls(
                path=base_path,
                batch_type="unsupervised" if base_name == "dsprites" else None,
            )
            dataset = ColoredMultiShapeDataset(
                base_dataset=base_ds,
                n_shapes=multi_n_shapes,
                num_colors=multi_num_colors,
                prediction_type="unsupervised",
                seed=int(seed),
            )
        else:
            if filter_expr.strip():
                train_ds, _ = get_cached_dataset(dataset_type, filter_expr.strip())
                dataset = train_ds
            else:
                cls, path = DATASETS[dataset_type]
                dataset = cls(path, data_filter=None)

        st.info(
            f"Dataset: **{dataset_type}** | Total items: **{len(dataset):,}** | Factors: `{getattr(dataset, 'factors', 'N/A')}`"
        )
        grid_img = create_dataset_grid_image(
            dataset=dataset, num_samples=num_samples, seed=int(seed)
        )
        st.image(
            grid_img,
            caption=f"{dataset_type} ({num_samples} samples)",
            use_container_width=True,
        )

        if hasattr(dataset, "factors") and hasattr(dataset, "factor_values"):
            st.subheader("Factor Distributions")
            factor_names = list(dataset.factors)
            selected_factor = st.selectbox("Factor to inspect", factor_names, index=0)
            idx = factor_names.index(selected_factor)
            vals = dataset.factor_values[:, idx]
            fig = px.histogram(
                x=vals,
                nbins=30,
                labels={"x": selected_factor},
                title=f"Distribution of {selected_factor}",
            )
            fig.update_layout(
                template="plotly_white", height=300, margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig, use_container_width=True)


def render_latent_analysis():
    st.header("🌌 Latent Space Representation Explorer")
    st.markdown(
        "Load trained checkpoints, generate or inspect cached embeddings, and dynamically project latents via PCA or UMAP."
    )

    # Sidebar checkpoint selection
    st.sidebar.subheader("Model & Checkpoint")
    log_dir = Path("data/logs")
    available_ckpts = find_checkpoints(log_dir)

    ckpt_options = [c["checkpoint_path"].as_posix() for c in available_ckpts]
    custom_str = "Custom Path..."
    ckpt_select = st.sidebar.selectbox("Choose Checkpoint", ckpt_options + [custom_str])

    if ckpt_select == custom_str:
        ckpt_path_str = st.sidebar.text_input("Checkpoint Path (.ckpt)", value="")
    else:
        ckpt_path_str = ckpt_select

    if not ckpt_path_str or not Path(ckpt_path_str).exists():
        st.warning("Please select a valid model checkpoint from the sidebar.")
        return

    ckpt_path = Path(ckpt_path_str)
    embeddings_path = ckpt_path.parent / "embeddings.npz"

    # Infer model type and dataset
    default_model, default_ds = infer_model_and_dataset(ckpt_path)
    model_name = st.sidebar.selectbox(
        "Model Type",
        list(MODEL_TYPES.keys()),
        index=list(MODEL_TYPES.keys()).index(default_model),
    )
    dataset_name = st.sidebar.selectbox(
        "Dataset", list(DATASETS.keys()), index=list(DATASETS.keys()).index(default_ds)
    )

    default_filter = "( shape == 3 ) & ( posX < 0.50 )"
    if "rotation" in ckpt_path.as_posix():
        if dataset_name == "pentominos":
            default_filter = "np.isin( shape , [1, 3, 5, 8] ) & ( angle >= 180 )"
        else:
            default_filter = "( shape == 3 ) & ( orientation < np.pi )"

    filter_expr = st.sidebar.text_input("Split Filter Expression", value=default_filter)
    include_test = st.sidebar.checkbox("Include Held-out Test Split", value=True)

    # Check embeddings cache
    train_emb, test_emb = load_cached_embeddings(embeddings_path)
    has_embeddings = train_emb is not None

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Plot Subsampling")
    max_pts = len(train_emb) if train_emb is not None else 50000
    sample_size = st.sidebar.slider(
        "Subsample Points for Plotting",
        min_value=500,
        max_value=min(50000, max_pts),
        value=min(2500, max_pts),
        step=500,
        help="Subsample projected coordinates for fast, high-framerate WebGL rendering (default: 2,500).",
    )

    status_col1, status_col2 = st.columns([2, 1])
    with status_col1:
        if has_embeddings:
            test_shape_str = f", Test: {test_emb.shape}" if test_emb is not None else ""
            st.success(
                f"✅ Cached embeddings loaded from: `{embeddings_path}` (Train: {train_emb.shape}{test_shape_str})"
            )
        else:
            st.warning(
                f"⚠️ No embeddings cache found at `{embeddings_path}`. Compute them below."
            )

    with status_col2:
        compute_btn = st.button(
            "⚡ Compute & Save Embeddings"
            if not has_embeddings
            else "🔄 Recompute Embeddings"
        )

    if compute_btn:
        progress_bar = st.progress(0.0, text="Initializing dataset & dataloaders...")
        train_dataset, test_dataset, train_loader, test_loader = load_dataset(
            dataset_name=dataset_name,
            filter_expr=filter_expr,
            batch_size=64,
            num_workers=0,
        )
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = get_cached_model(
            model_name=model_name, checkpoint_path=str(ckpt_path), device=device
        )

        total_splits = 2 if include_test else 1

        def update_train_progress(current, total):
            frac = (current / total) / total_splits
            progress_bar.progress(
                frac,
                text=f"Computing train embeddings: batch {current}/{total} ({int(frac * 100)}%)",
            )

        train_emb = compute_embeddings(
            model=model,
            loader=train_loader,
            device=torch.device(device),
            progress_callback=update_train_progress,
        )

        test_emb = None
        if include_test:

            def update_test_progress(current, total):
                frac = 0.5 + (current / total) * 0.5
                progress_bar.progress(
                    frac,
                    text=f"Computing test embeddings: batch {current}/{total} ({int(frac * 100)}%)",
                )

            test_emb = compute_embeddings(
                model=model,
                loader=test_loader,
                device=torch.device(device),
                progress_callback=update_test_progress,
            )

        progress_bar.progress(1.0, text="Saving embeddings to disk...")
        save_dict = {"train_embeddings": train_emb}
        if test_emb is not None:
            save_dict["test_embeddings"] = test_emb
        np.savez(embeddings_path, **save_dict)
        progress_bar.empty()
        st.success(f"Saved {len(train_emb)} train embeddings to `{embeddings_path}`!")
        st.rerun()

    if train_emb is None:
        st.stop()

    # Load dataset factors for coloring
    train_ds, test_ds = get_cached_dataset(dataset_name=dataset_name, filter_expr=filter_expr)

    train_factors = train_ds.factor_values
    test_factors = test_ds.factor_values if test_emb is not None else None
    factor_names = list(type(train_ds).factors)

    st.markdown("---")
    st.subheader("⚙️ Dimensionality Reduction Settings")
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        method = st.radio("Reduction Method", ["PCA", "UMAP"], horizontal=True)

    with ctrl_col2:
        is_3d = st.radio("Dimensionality", ["3D", "2D"], horizontal=True) == "3D"

    n_components = 3 if is_3d else 2
    n_neighbors = 15
    min_dist = 0.1
    metric = "euclidean"
    compute_metrics = False

    if method == "UMAP":
        with st.expander("⚙️ UMAP Hyperparameters", expanded=True):
            u_col1, u_col2, u_col3, u_col4 = st.columns(4)
            n_neighbors = u_col1.slider("n_neighbors", 5, 50, 15)
            min_dist = u_col2.slider("min_dist", 0.01, 0.99, 0.1)
            metric = u_col3.selectbox(
                "Metric", ["euclidean", "cosine", "manhattan"], index=0
            )
            compute_metrics = u_col4.checkbox(
                "Compute Trustworthiness & Continuity", value=False
            )

    if method == "PCA":
        cache_key = f"pca_{ckpt_path}_{is_3d}_{len(train_emb)}_{len(test_emb) if test_emb is not None else 0}"
    else:
        cache_key = f"umap_{ckpt_path}_{is_3d}_{n_neighbors}_{min_dist}_{metric}_{compute_metrics}_{len(train_emb)}_{len(test_emb) if test_emb is not None else 0}"

    is_computed = cache_key in st.session_state

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        btn_label = (
            f"🚀 Compute {method} Projection"
            if not is_computed
            else f"🔄 Recompute {method} Projection"
        )
        compute_proj_btn = st.button(btn_label, type="primary")

    with btn_col2:
        cancel_btn = st.button("⏹️ Cancel Running Process")
        if cancel_btn:
            cancel_active_reduction_process()
            st.info("Process cancellation signal delivered.")

    if compute_proj_btn:
        cancel_active_reduction_process()
        prog_bar = st.progress(
            0.0, text=f"Launching {method} in isolated subprocess..."
        )

        kwargs = {
            "train_embeddings": train_emb,
            "test_embeddings": test_emb,
            "n_components": n_components,
        }
        if method == "UMAP":
            kwargs.update(
                {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                    "compute_metrics": compute_metrics,
                }
            )

        def update_progress(frac, text):
            prog_bar.progress(min(1.0, max(0.0, frac)), text=text)

        with st.spinner(f"Running {method} on full dataset in isolated subprocess..."):
            results = run_reduction_in_process(
                method=method,
                kwargs=kwargs,
                progress_callback=update_progress,
            )
        st.session_state[cache_key] = results
        prog_bar.empty()
        st.success(f"✅ {method} projection completed successfully!")
        st.rerun()

    if not is_computed:
        pts_str = f" ({len(train_emb):,} points)" if train_emb is not None else ""
        st.info(
            f"👉 Parameters selected for **{method}** ({'3D' if is_3d else '2D'}). "
            f"Click **'{btn_label}'** above to run dimensionality reduction on the full dataset{pts_str}. "
            f"Once computed, **{sample_size:,}** points will be rendered for fast, responsive visualization."
        )
        return

    results = st.session_state[cache_key]
    train_proj = results["train_proj"]
    test_proj = results["test_proj"]

    if method == "PCA":
        with st.expander("📊 PCA Explained Variance", expanded=False):
            var_fig = create_pca_variance_plot(results["explained_variance_ratio"])
            st.plotly_chart(var_fig, use_container_width=True)
    elif method == "UMAP" and compute_metrics and results.get("metrics") is not None:
        with st.expander("📈 UMAP Quality Metrics", expanded=False):
            u_fig = create_umap_metrics_plot(results["metrics"])
            st.plotly_chart(u_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎨 Plotting Controls")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        shade_options = factor_names + [
            f"Latent z_{i}" for i in range(min(16, train_emb.shape[1]))
        ]
        selected_shade = st.selectbox("Color / Shade by", shade_options, index=0)

    # Proportional subsampling across train and held-out test splits
    total_available = len(train_proj) + (len(test_proj) if test_proj is not None else 0)
    train_budget = (
        int(sample_size * (len(train_proj) / total_available))
        if total_available > 0
        else sample_size
    )
    train_budget = max(1, min(train_budget, len(train_proj)))

    train_proj_sub, train_factors_sub, train_idx = subsample_points(
        train_proj, train_factors, train_budget, seed=42
    )
    test_proj_sub = None
    test_factors_sub = None
    test_idx = None
    if test_proj is not None and test_factors is not None and len(test_proj) > 0:
        test_budget = min(len(test_proj), max(0, sample_size - train_budget))
        if test_budget > 0:
            test_proj_sub, test_factors_sub, test_idx = subsample_points(
                test_proj, test_factors, test_budget, seed=42
            )

    actual_plotted = len(train_proj_sub) + (
        len(test_proj_sub) if test_proj_sub is not None else 0
    )

    with plot_col2:
        st.metric(
            label="Rendered Data Points",
            value=f"{actual_plotted:,}",
            delta=(
                f"-{(total_available - actual_plotted):,} subsampled"
                if total_available > actual_plotted
                else "full data"
            ),
        )
        st.caption(
            "Adjust sample size anytime in the left sidebar under 'Plot Subsampling'."
        )

    test_info = f" | Test: {len(test_proj_sub):,}" if test_proj_sub is not None else ""
    st.info(
        f"📍 **Plotting {actual_plotted:,} points** (subsampled from {total_available:,} total computed coordinates) — "
        f"Train: {len(train_proj_sub):,}{test_info}. Adjust sample size in the sidebar for denser or lighter rendering."
    )

    is_categorical = False
    label_map = None

    if selected_shade in factor_names:
        factor_idx = factor_names.index(selected_shade)
        train_color = train_factors_sub[:, factor_idx]
        test_color = (
            test_factors_sub[:, factor_idx] if test_factors_sub is not None else None
        )
        if selected_shade == "shape" and hasattr(train_ds, "map_shapes"):
            is_categorical = True
            label_map = train_ds.map_shapes
    else:
        # Latent dimension
        latent_dim = int(selected_shade.split("_")[1])
        train_color = train_emb[train_idx, latent_dim]
        test_color = (
            test_emb[test_idx, latent_dim]
            if test_emb is not None and test_idx is not None
            else None
        )

    # Style controls
    style_col1, style_col2, style_col3 = st.columns([1, 1, 2])
    marker_size = style_col1.slider("Point Size", min_value=1, max_value=10, value=3)
    opacity = style_col2.slider(
        "Opacity", min_value=0.1, max_value=1.0, value=0.4, step=0.05
    )
    colorscale = style_col3.selectbox(
        "Color Palette",
        ["Viridis", "Plasma", "Turbo", "Cividis", "Rainbow", "Blues", "Inferno"],
    )

    scatter_fig = create_latent_scatter(
        train_proj=train_proj_sub,
        train_color=train_color,
        test_proj=test_proj_sub,
        test_color=test_color,
        factor_name=selected_shade,
        is_categorical=is_categorical,
        label_map=label_map,
        is_3d=is_3d,
        marker_size=marker_size,
        opacity=opacity,
        colorscale=colorscale,
        title=f"{method} Latent Space ({actual_plotted:,} pts, {'3D' if is_3d else '2D'}) - {selected_shade}",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)


def render_reconstructions():
    st.header("🔍 Model Reconstruction Quality")
    st.markdown(
        "Visually compare ground-truth inputs side-by-side with model reconstructions."
    )

    log_dir = Path("data/logs")
    available_ckpts = find_checkpoints(log_dir)
    ckpt_options = [c["checkpoint_path"].as_posix() for c in available_ckpts]

    col1, col2 = st.columns([1, 2])
    with col1:
        ckpt_str = st.selectbox("Checkpoint", ckpt_options)
        ckpt_path = Path(ckpt_str)

        default_model, default_ds = infer_model_and_dataset(ckpt_path)
        model_name = st.selectbox(
            "Model Type",
            list(MODEL_TYPES.keys()),
            index=list(MODEL_TYPES.keys()).index(default_model),
        )
        dataset_name = st.selectbox(
            "Dataset",
            list(DATASETS.keys()),
            index=list(DATASETS.keys()).index(default_ds),
        )

        filter_expr = st.text_input(
            "Filter Expression", value="( shape == 3 ) & ( posX < 0.50 )"
        )
        num_samples = st.slider("Number of pairs", min_value=2, max_value=16, value=8)
        seed = st.number_input("Seed", value=42, step=1)
        split = st.radio("Data Split", ["Test / Held-out Split", "Training Split"])

    with col2:
        train_ds, test_ds = get_cached_dataset(
            dataset_name=dataset_name, filter_expr=filter_expr
        )
        target_ds = test_ds if split.startswith("Test") else train_ds

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = get_cached_model(
            model_name=model_name, checkpoint_path=str(ckpt_path), device=device
        )

        st.subheader("Originals (Left) vs Reconstructions (Right)")
        recon_img = create_reconstruction_grid_image(
            model=model,
            dataset=target_ds,
            num_samples=num_samples,
            seed=int(seed),
            device=device,
        )
        st.image(
            recon_img,
            caption="Alternating pairs: [Original, Reconstruction]",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
