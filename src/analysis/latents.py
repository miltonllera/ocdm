import glob
import time
import multiprocessing as mp
from pathlib import Path
from functools import partial
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import torch

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_model,
    load_dataset,
    compute_embeddings,
    load_or_compute_embeddings,
)


def bootstrap_score(score_fn, x: np.ndarray, z: np.ndarray, n_bootstrap: int = 10, sample_size: int = 10000):
    scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
        scores.append(score_fn(x[idx], z[idx], n_neighbors=15))
    return float(np.mean(scores)), float(np.var(scores))


def find_checkpoints(log_dir: Path):
    ckpts = sorted(glob.glob(str(log_dir / "**" / "*.ckpt"), recursive=True))
    results = []
    for ckpt in ckpts:
        ckpt_path = Path(ckpt)
        emb_path = ckpt_path.parent / "embeddings.npz"
        results.append({
            "checkpoint_path": ckpt_path,
            "embeddings_path": emb_path,
            "has_embeddings": emb_path.exists(),
            "name": ckpt_path.parent.name,
        })
    return results


def subsample_points(data: np.ndarray, factors: np.ndarray, sample_size: int, seed: int = 42):
    if len(data) <= sample_size:
        return data, factors, np.arange(len(data))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=sample_size, replace=False)
    return data[idx], factors[idx], idx


class TqdmProgressWriter:
    def __init__(self, callback):
        self.callback = callback

    def write(self, s):
        import re
        s_str = str(s).strip()
        if not s_str:
            return
        m = re.search(r'(\d+)%', s_str)
        if m:
            pct = int(m.group(1))
            frac = 0.2 + 0.7 * (pct / 100.0)
            self.callback(frac, f"Optimizing UMAP embedding ({pct}%)...")
        elif "Finding Nearest Neighbors" in s_str:
            self.callback(0.05, "Finding nearest neighbors...")
        elif "Construct fuzzy simplicial set" in s_str:
            self.callback(0.12, "Constructing fuzzy simplicial set graph...")
        elif "Construct embedding" in s_str:
            self.callback(0.20, "Optimizing low-dimensional layout...")
        elif "completed" in s_str:
            m_ep = re.search(r'completed\s+(\d+)\s+/\s+(\d+)\s+epochs', s_str)
            if m_ep:
                cur, tot = int(m_ep.group(1)), int(m_ep.group(2))
                frac = 0.2 + 0.7 * (cur / max(1, tot))
                self.callback(frac, f"Optimizing UMAP layout: epoch {cur}/{tot} ({int(frac*100)}%)...")

    def flush(self):
        pass


def run_pca_reduction(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray | None,
    n_components: int = 3,
    progress_callback=None,
):
    if progress_callback is not None:
        progress_callback(0.1, "Fitting PCA on full dataset...")

    if test_embeddings is not None:
        all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    else:
        all_embeddings = train_embeddings

    pca = PCA()
    pca.fit(all_embeddings)

    if progress_callback is not None:
        progress_callback(0.7, "Transforming coordinates...")

    train_proj = pca.transform(train_embeddings)[:, :n_components]
    test_proj = pca.transform(test_embeddings)[:, :n_components] if test_embeddings is not None else None

    if progress_callback is not None:
        progress_callback(1.0, "PCA complete!")

    return {
        "pca": pca,
        "train_proj": train_proj,
        "test_proj": test_proj,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }


def run_umap_reduction(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray | None,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    compute_metrics: bool = True,
    random_state: int | None = None,
    progress_callback=None,
):
    tqdm_kwds = None
    verbose = False
    if progress_callback is not None:
        tqdm_kwds = {"file": TqdmProgressWriter(progress_callback)}
        verbose = True
        progress_callback(0.02, "Initializing UMAP on full dataset...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1 if random_state is None else 1,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
    )

    if test_embeddings is not None:
        all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    else:
        all_embeddings = train_embeddings

    proj = reducer.fit_transform(all_embeddings)

    metrics = None
    if compute_metrics:
        if progress_callback is not None:
            progress_callback(0.92, "Computing Trustworthiness & Continuity metrics...")
        t_mean, t_var = bootstrap_score(partial(trustworthiness, metric=metric), all_embeddings, proj)
        c_mean, c_var = bootstrap_score(partial(trustworthiness, metric=metric), proj, all_embeddings)
        metrics = {
            "trustworthiness_mean": t_mean,
            "trustworthiness_var": t_var,
            "continuity_mean": c_mean,
            "continuity_var": c_var,
        }

    reducer.tqdm_kwds = None

    if test_embeddings is not None:
        train_proj = proj[:len(train_embeddings)]
        test_proj = proj[len(train_embeddings):]
    else:
        train_proj = proj
        test_proj = None

    if progress_callback is not None:
        progress_callback(1.0, "UMAP projection complete!")

    return {
        "reducer": reducer,
        "train_proj": train_proj,
        "test_proj": test_proj,
        "metrics": metrics,
    }


def load_cached_embeddings(embeddings_path: Path):
    if not embeddings_path.exists():
        return None, None
    data = np.load(embeddings_path)
    train_emb = data["train_embeddings"].squeeze() if "train_embeddings" in data else None
    test_emb = data["test_embeddings"].squeeze() if "test_embeddings" in data else None
    return train_emb, test_emb


def compute_and_save_latents(
    model_name: str,
    checkpoint_path: Path,
    dataset_name: str,
    filter_expr: str,
    batch_size: int = 64,
    include_test: bool = True,
    device: str | None = None,
):
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name=dataset_name,
        filter_expr=filter_expr,
        batch_size=batch_size,
        num_workers=0,
    )

    model, model_device = load_model(model_name=model_name, checkpoint_path=checkpoint_path, compile=False)
    if device is not None:
        model = model.to(torch.device(device))
        model_device = torch.device(device)

    train_emb = compute_embeddings(model=model, loader=train_loader, device=model_device)
    test_emb = None
    if include_test:
        test_emb = compute_embeddings(model=model, loader=test_loader, device=model_device)

    embeddings_path = checkpoint_path.parent / "embeddings.npz"
    save_dict = {"train_embeddings": train_emb}
    if test_emb is not None:
        save_dict["test_embeddings"] = test_emb
    np.savez(embeddings_path, **save_dict)

    return train_emb, test_emb, train_dataset, test_dataset


_ACTIVE_PROCESS = None


def cancel_active_reduction_process():
    global _ACTIVE_PROCESS
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.is_alive():
        _ACTIVE_PROCESS.terminate()
        _ACTIVE_PROCESS.join(timeout=1.0)
        if _ACTIVE_PROCESS.is_alive():
            _ACTIVE_PROCESS.kill()
        _ACTIVE_PROCESS = None


def _reduction_process_worker(out_queue, prog_queue, method, kwargs):
    try:
        def on_progress(frac, text):
            prog_queue.put((frac, text))

        if method == "PCA":
            res = run_pca_reduction(**kwargs, progress_callback=on_progress)
        else:
            res = run_umap_reduction(**kwargs, progress_callback=on_progress)

        out_queue.put(("SUCCESS", res))
    except Exception as e:
        out_queue.put(("ERROR", str(e)))


def run_reduction_in_process(method: str, kwargs: dict, progress_callback=None, poll_interval: float = 0.1):
    global _ACTIVE_PROCESS
    cancel_active_reduction_process()

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    prog_queue = ctx.Queue()

    proc = ctx.Process(
        target=_reduction_process_worker,
        args=(out_queue, prog_queue, method, kwargs),
    )
    _ACTIVE_PROCESS = proc
    proc.start()

    try:
        while proc.is_alive():
            while not prog_queue.empty():
                try:
                    frac, text = prog_queue.get_nowait()
                    if progress_callback is not None:
                        progress_callback(frac, text)
                except Exception:
                    break
            time.sleep(poll_interval)

        while not prog_queue.empty():
            try:
                frac, text = prog_queue.get_nowait()
                if progress_callback is not None:
                    progress_callback(frac, text)
            except Exception:
                break

        if not out_queue.empty():
            status, payload = out_queue.get()
            if status == "SUCCESS":
                return payload
            else:
                raise RuntimeError(f"Error in {method} process: {payload}")
        else:
            raise RuntimeError(f"{method} process terminated prematurely with exit code {proc.exitcode}")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
        _ACTIVE_PROCESS = None
