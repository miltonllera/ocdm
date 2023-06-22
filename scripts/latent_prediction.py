import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_or_compute_embeddings,
    load_model,
    load_dataset
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def cross_validate_regression(embeddings, targets, n_splits=5, n_repeats=10):
    scores = []
    kfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for train_idx, val_idx in kfold.split(embeddings):
        x_train, y_train = embeddings[train_idx], targets[train_idx]
        x_val, y_val = embeddings[val_idx], targets[val_idx]

        model = LinearRegression().fit(x_train, y_train)
        scores.append(model.score(x_val, y_val))

    return np.mean(scores), np.std(scores)


def evaluate_on_test(train_embeddings, train_targets, test_embeddings, test_targets):
    model = LinearRegression().fit(train_embeddings, train_targets)
    return model.score(test_embeddings, test_targets)


def run_factor_prediction(
    train_embeddings, train_factors, test_embeddings, test_factors,
    factor_names, conditioning_factor_idx=0
):
    conditioning_factor_name = factor_names[conditioning_factor_idx]
    results = {
        'unconditional': {},
        'conditioned': {}
    }

    unique_cond_values = np.unique(train_factors[:, conditioning_factor_idx])

    _logger.info("Running unconditional factor prediction...")
    for i, factor_name in enumerate(factor_names):
        _logger.info(f"  Factor: {factor_name}")

        cv_mean, cv_std = cross_validate_regression(
            train_embeddings, train_factors[:, i]
        )

        test_score = evaluate_on_test(
            train_embeddings, train_factors[:, i],
            test_embeddings, test_factors[:, i]
        )

        results['unconditional'][factor_name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_score': test_score
        }

        _logger.info(f"    CV: {cv_mean:.4f} +/- {cv_std:.4f}, Test: {test_score:.4f}")

    _logger.info(f"Running prediction conditioned on {conditioning_factor_name}...")
    for cond_val in unique_cond_values:
        cond_name = f"{conditioning_factor_name}_{int(cond_val)}"
        results['conditioned'][cond_name] = {}

        train_mask = train_factors[:, conditioning_factor_idx] == cond_val
        test_mask = test_factors[:, conditioning_factor_idx] == cond_val

        train_emb_cond = train_embeddings[train_mask]
        train_fac_cond = train_factors[train_mask]
        test_emb_cond = test_embeddings[test_mask]
        test_fac_cond = test_factors[test_mask]

        if len(train_emb_cond) == 0:
            _logger.warning(
                f"  No training samples for {conditioning_factor_name}={int(cond_val)}"
            )
            continue

        _logger.info(
            f"  {conditioning_factor_name}={int(cond_val)} "
            f"(train: {len(train_emb_cond)}, test: {len(test_emb_cond)})"
        )

        for i, factor_name in enumerate(factor_names):
            if i == conditioning_factor_idx:
                continue

            cv_mean, cv_std = cross_validate_regression(
                train_emb_cond, train_fac_cond[:, i]
            )

            if len(test_emb_cond) == 0:
                _logger.info(
                    f"  No test samples for {conditioning_factor_name}={int(cond_val)}"
                )
                test_score = 0.0

            else:
                test_score = evaluate_on_test(
                    train_emb_cond, train_fac_cond[:, i],
                    test_emb_cond, test_fac_cond[:, i]
                )

            results['conditioned'][cond_name][factor_name] = {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_score': test_score
            }

            _logger.info(
                f"    {factor_name}: CV: {cv_mean:.4f} +/- {cv_std:.4f}, Test: {test_score:.4f}"
            )

    return results


def save_results_csv(output_path, results, factor_names):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Unconditional Factor Prediction"])
        writer.writerow(["Factor", "CV Mean", "CV Std", "Test Score"])
        for factor_name in factor_names:
            if factor_name in results['unconditional']:
                r = results['unconditional'][factor_name]
                writer.writerow([factor_name, r['cv_mean'], r['cv_std'], r['test_score']])

        writer.writerow([])

        writer.writerow(["Conditioned Factor Prediction"])
        for cond_name, cond_results in results['conditioned'].items():
            writer.writerow([cond_name])
            writer.writerow(["Factor", "CV Mean", "CV Std", "Test Score"])
            for factor_name, r in cond_results.items():
                writer.writerow([factor_name, r['cv_mean'], r['cv_std'], r['test_score']])
            writer.writerow([])

    _logger.info(f"Results saved to {output_path}")


def print_results_summary(results, factor_names):
    print("\n" + "=" * 60)
    print("UNCONDITIONAL FACTOR PREDICTION")
    print("=" * 60)
    print(f"{'Factor':<15} {'CV Mean':>10} {'CV Std':>10} {'Test':>10}")
    print("-" * 60)
    for factor_name in factor_names:
        if factor_name in results['unconditional']:
            r = results['unconditional'][factor_name]
            print(f"{factor_name:<15} {r['cv_mean']:>10.4f} {r['cv_std']:>10.4f} {r['test_score']:>10.4f}")

    print("\n" + "=" * 60)
    print("CONDITIONED FACTOR PREDICTION")
    print("=" * 60)
    for cond_name, cond_results in results['conditioned'].items():
        print(f"\n{cond_name}:")
        print(f"{'Factor':<15} {'CV Mean':>10} {'CV Std':>10} {'Test':>10}")
        print("-" * 45)
        for factor_name, r in cond_results.items():
            print(f"{factor_name:<15} {r['cv_mean']:>10.4f} {r['cv_std']:>10.4f} {r['test_score']:>10.4f}")


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    output_dir: Path,
    batch_size: int = 64,
    force_recompute: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset_name}"
    embeddings_path = model_checkpoint.parent / "embeddings.npz"

    _logger.info("Loading datasets...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size
    )

    train_factors = train_dataset.factor_values
    test_factors = test_dataset.factor_values
    factor_names = type(train_dataset).factors

    _logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    _logger.info("Loading model...")
    model, device = load_model(model_name, model_checkpoint)

    train_embeddings = load_or_compute_embeddings(
        embeddings_path, model, train_loader, device, force_recompute, "train"
    )
    test_embeddings = load_or_compute_embeddings(
        embeddings_path, model, test_loader, device, force_recompute, "test"
    )

    _logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    _logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    results = run_factor_prediction(
        train_embeddings, train_factors,
        test_embeddings, test_factors,
        factor_names
    )

    csv_path = output_dir / f"{prefix}_prediction_results.csv"
    save_results_csv(csv_path, results, factor_names)

    print_results_summary(results, factor_names)

    _logger.info("Analysis complete!")


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
    parser.add_argument("--output_dir", type=Path, default=Path("plots/latent_prediction"),
        help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Batch size for embedding computation")
    parser.add_argument("--force_recompute", action="store_true",
        help="Force recomputation of embeddings even if cached file exists")

    args = parser.parse_args()
    main(**vars(args))
