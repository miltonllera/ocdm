import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.stats import chi2
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResults

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    load_or_compute_embeddings,
    load_model,
    load_dataset,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def resolve_conditioning_factor(factor_names: list[str], conditioning_factor: str) -> int:
    for i, name in enumerate(factor_names):
        if name == conditioning_factor:
            return i
    raise ValueError(
        f"Conditioning factor '{conditioning_factor}' not found in factor names: {factor_names}"
    )


def build_mixed_effects_df(
    embeddings: np.ndarray,
    factor_values: np.ndarray,
    factor_idx: int,
    conditioning_idx: int,
) -> pd.DataFrame:
    d = embeddings.shape[1]
    z_columns = [f"z_{j}" for j in range(d)]

    data = {col: embeddings[:, j] for j, col in enumerate(z_columns)}
    data["y"] = factor_values[:, factor_idx]
    data["group"] = factor_values[:, conditioning_idx].astype(str)

    return pd.DataFrame(data)


def fit_random_intercepts(df: pd.DataFrame, z_columns: list[str]) -> MixedLMResults:
    exog = sm.add_constant(df[z_columns])
    model = MixedLM(endog=df["y"], exog=exog, groups=df["group"])
    result = model.fit(reml=False, method='powell', maxiter=200)
    return result


def fit_random_slopes(
    df: pd.DataFrame, z_columns: list[str]
) -> MixedLMResults | None:
    exog = sm.add_constant(df[z_columns])
    exog_re = sm.add_constant(df[z_columns])
    model = MixedLM(endog=df["y"], exog=exog, groups=df["group"], exog_re=exog_re)
    try:
        result = model.fit(reml=False, method='powell', maxiter=200)
    except (LinAlgError, ValueError) as e:
        _logger.warning(f"Random slopes model failed to fit: {e}")
        return None
    if not result.converged:
        _logger.warning("Random slopes model did not converge")
        return None
    return result


def compare_models(
    ri_result: MixedLMResults, rs_result: MixedLMResults | None
) -> dict:
    comparison = {
        "rs_converged": rs_result is not None,
        "rs_aic": None,
        "rs_bic": None,
        "rs_llf": None,
        "lrt_statistic": None,
        "lrt_df": None,
        "lrt_pvalue": None,
    }
    if rs_result is None:
        return comparison

    comparison["rs_aic"] = rs_result.aic
    comparison["rs_bic"] = rs_result.bic
    comparison["rs_llf"] = rs_result.llf

    lrt_stat = -2.0 * (ri_result.llf - rs_result.llf)
    df_diff = rs_result.df_modelwc - ri_result.df_modelwc
    p_value = chi2.sf(lrt_stat, df_diff)

    comparison["lrt_statistic"] = lrt_stat
    comparison["lrt_df"] = df_diff
    comparison["lrt_pvalue"] = p_value

    return comparison


def compute_r_squared(
    result: MixedLMResults, df: pd.DataFrame, z_columns: list[str]
) -> tuple[float, float]:
    ss_total = np.sum((df["y"].values - df["y"].mean()) ** 2)

    exog = sm.add_constant(df[z_columns])
    fixed_pred = exog.values @ result.fe_params.values
    ss_res_marginal = np.sum((df["y"].values - fixed_pred) ** 2)
    r2_marginal = 1.0 - ss_res_marginal / ss_total

    fitted = result.fittedvalues.values
    ss_res_conditional = np.sum((df["y"].values - fitted) ** 2)
    r2_conditional = 1.0 - ss_res_conditional / ss_total

    return r2_marginal, r2_conditional


def predict_test(
    result: MixedLMResults,
    test_df: pd.DataFrame,
    z_columns: list[str],
    has_random_slopes: bool,
) -> tuple[float, float]:
    ss_total = np.sum((test_df["y"].values - test_df["y"].mean()) ** 2)

    exog = sm.add_constant(test_df[z_columns])
    fixed_pred = exog.values @ result.fe_params.values
    ss_res_marginal = np.sum((test_df["y"].values - fixed_pred) ** 2)
    r2_marginal = 1.0 - ss_res_marginal / ss_total

    random_effects = result.random_effects
    conditional_pred = fixed_pred.copy()
    for group_name, re_values in random_effects.items():
        mask = test_df["group"].values == group_name
        if not np.any(mask):
            continue
        if has_random_slopes:
            group_exog_re = sm.add_constant(test_df.loc[mask, z_columns])
            conditional_pred[mask] += group_exog_re.values @ re_values.values
        else:
            conditional_pred[mask] += re_values.values[0]

    ss_res_conditional = np.sum((test_df["y"].values - conditional_pred) ** 2)
    r2_conditional = 1.0 - ss_res_conditional / ss_total

    return r2_marginal, r2_conditional


def run_mixed_effects(
    train_embeddings: np.ndarray,
    train_factors: np.ndarray,
    test_embeddings: np.ndarray,
    test_factors: np.ndarray,
    factor_names: list[str],
    conditioning_idx: int,
    random_slopes: bool,
    factor_indices: list[int] | None,
) -> list[dict]:
    conditioning_name = factor_names[conditioning_idx]
    d = train_embeddings.shape[1]
    z_columns = [f"z_{j}" for j in range(d)]
    results = []

    if factor_indices is None:
        factor_indices = [i for i in range(len(factor_names)) if i != conditioning_idx]

    for i in factor_indices:
        factor_name = factor_names[i]

        _logger.info(f"Fitting mixed effects models for factor: {factor_name}")

        train_df = build_mixed_effects_df(
            train_embeddings, train_factors, i, conditioning_idx,
        )
        test_df = build_mixed_effects_df(
            test_embeddings, test_factors, i, conditioning_idx,
        )

        n_groups = train_df["group"].nunique()
        _logger.info(f"  Groups: {n_groups}, Train: {len(train_df)}, Test: {len(test_df)}")

        _logger.info("  Fitting random intercepts model...")
        ri_result = fit_random_intercepts(train_df, z_columns)
        ri_r2_train = compute_r_squared(ri_result, train_df, z_columns)
        ri_r2_test = predict_test(ri_result, test_df, z_columns, has_random_slopes=False)
        _logger.info(
            f"  RI train R²: marginal={ri_r2_train[0]:.4f}, conditional={ri_r2_train[1]:.4f}"
        )
        _logger.info(
            f"  RI test  R²: marginal={ri_r2_test[0]:.4f}, conditional={ri_r2_test[1]:.4f}"
        )

        row = {
            "factor": factor_name,
            "conditioning_factor": conditioning_name,
            "n_groups": n_groups,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "ri_r2_marginal_train": ri_r2_train[0],
            "ri_r2_conditional_train": ri_r2_train[1],
            "ri_r2_marginal_test": ri_r2_test[0],
            "ri_r2_conditional_test": ri_r2_test[1],
            "ri_aic": ri_result.aic,
            "ri_bic": ri_result.bic,
            "ri_llf": ri_result.llf,
            "fe_const": ri_result.fe_params.values[0],
            **{f"fe_z_{j}": ri_result.fe_params.values[j + 1] for j in range(d)},
            "re_var_intercept": ri_result.cov_re.iloc[0, 0],
        }

        if random_slopes:
            _logger.info("  Fitting random slopes model...")
            rs_result = fit_random_slopes(train_df, z_columns)
            rs_r2_train = (None, None)
            rs_r2_test = (None, None)
            if rs_result is not None:
                rs_r2_train = compute_r_squared(rs_result, train_df, z_columns)
                rs_r2_test = predict_test(rs_result, test_df, z_columns, has_random_slopes=True)
                _logger.info(
                    f"  RS train R²: marginal={rs_r2_train[0]:.4f}, conditional={rs_r2_train[1]:.4f}"
                )
                _logger.info(
                    f"  RS test  R²: marginal={rs_r2_test[0]:.4f}, conditional={rs_r2_test[1]:.4f}"
                )
            else:
                _logger.info("  RS model: FAILED")

            comparison = compare_models(ri_result, rs_result)
            row.update({
                "rs_r2_marginal_train": rs_r2_train[0],
                "rs_r2_conditional_train": rs_r2_train[1],
                "rs_r2_marginal_test": rs_r2_test[0],
                "rs_r2_conditional_test": rs_r2_test[1],
                **comparison,
            })

        results.append(row)

    return results


def save_results_csv(output_path: Path, results: list[dict]) -> None:
    if not results:
        _logger.warning("No results to save")
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    _logger.info(f"Results saved to {output_path}")


def print_results_summary(
    results: list[dict], conditioning_factor: str, random_slopes: bool
) -> None:
    print("\n" + "=" * 100)
    print(f"MIXED EFFECTS RESULTS (conditioning on: {conditioning_factor})")
    print("=" * 100)

    if random_slopes:
        header = (
            f"{'Factor':<14} "
            f"{'RI R²m(tr)':>10} {'RI R²c(tr)':>10} {'RI R²m(te)':>10} {'RI R²c(te)':>10} "
            f"{'RS R²c(te)':>10} "
            f"{'RI AIC':>10} {'RS AIC':>10} "
            f"{'LRT p':>10}"
        )
        print(header)
        print("-" * 100)

        for r in results:
            rs_r2 = f"{r['rs_r2_conditional_test']:.4f}" if r.get("rs_r2_conditional_test") is not None else "FAILED"
            rs_aic = f"{r['rs_aic']:.1f}" if r.get("rs_aic") is not None else "N/A"
            lrt_p = f"{r['lrt_pvalue']:.2e}" if r.get("lrt_pvalue") is not None else "N/A"

            print(
                f"{r['factor']:<14} "
                f"{r['ri_r2_marginal_train']:>10.4f} {r['ri_r2_conditional_train']:>10.4f} "
                f"{r['ri_r2_marginal_test']:>10.4f} {r['ri_r2_conditional_test']:>10.4f} "
                f"{rs_r2:>10} "
                f"{r['ri_aic']:>10.1f} {rs_aic:>10} "
                f"{lrt_p:>10}"
            )
    else:
        header = (
            f"{'Factor':<14} "
            f"{'RI R²m(tr)':>10} {'RI R²c(tr)':>10} {'RI R²m(te)':>10} {'RI R²c(te)':>10} "
            f"{'RI AIC':>10}"
        )
        print(header)
        print("-" * 70)

        for r in results:
            print(
                f"{r['factor']:<14} "
                f"{r['ri_r2_marginal_train']:>10.4f} {r['ri_r2_conditional_train']:>10.4f} "
                f"{r['ri_r2_marginal_test']:>10.4f} {r['ri_r2_conditional_test']:>10.4f} "
                f"{r['ri_aic']:>10.1f}"
            )

    print()


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    output_dir: Path,
    batch_size: int,
    force_recompute: bool,
    conditioning_factor: str,
    random_slopes: bool,
    factor_indices: list[int] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{dataset_name}"
    embeddings_path = model_checkpoint.parent / "embeddings.npz"

    _logger.info("Loading datasets...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size
    )

    train_factors = train_dataset.factor_values
    test_factors = test_dataset.factor_values
    factor_names = list(type(train_dataset).factors)

    _logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    conditioning_idx = resolve_conditioning_factor(factor_names, conditioning_factor)
    _logger.info(f"Conditioning factor: {conditioning_factor} (index {conditioning_idx})")

    _logger.info("Loading model...")
    model, device = load_model(model_name, model_checkpoint)

    train_embeddings = load_or_compute_embeddings(
        embeddings_path, model, train_loader, device, force_recompute, "train"
    )
    test_embeddings = load_or_compute_embeddings(
        embeddings_path, model, test_loader, device, force_recompute, "test"
    )

    if train_embeddings.ndim == 3:
        _logger.info(
            f"Flattening 3D embeddings: {train_embeddings.shape} -> "
            f"({train_embeddings.shape[0]}, {train_embeddings.shape[1] * train_embeddings.shape[2]})"
        )
        train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)

    _logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    _logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    results = run_mixed_effects(
        train_embeddings, train_factors,
        test_embeddings, test_factors,
        factor_names, conditioning_idx, random_slopes, factor_indices,
    )

    csv_path = output_dir / f"{prefix}_mixed_effects.csv"
    save_results_csv(csv_path, results)

    print_results_summary(results, conditioning_factor, random_slopes)

    _logger.info("Analysis complete!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Mixed effects model for latent factor prediction")

    parser.add_argument(
        "--dataset_name", type=str, required=True,
        choices=list(DATASETS.keys()),
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--filter_expr", type=str, required=True,
        help="Filter expression to determine train/test split",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        choices=list(MODEL_TYPES.keys()),
        help="Type of model",
    )
    parser.add_argument(
        "--model_checkpoint", type=Path, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("plots/mixed_effects"),
        help="Directory to save output files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Force recomputation of embeddings even if cached file exists",
    )
    parser.add_argument(
        "--conditioning_factor", type=str, default="shape",
        help="Factor to use as the grouping variable (random effect)",
    )
    parser.add_argument(
        "--no_random_slopes", action="store_true",
        help="Skip fitting the random slopes model",
    )
    parser.add_argument(
        "--factor_indices", type=int, nargs="+", default=None,
        help="Indices of factors to predict (default: all non-conditioning factors)",
    )

    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        filter_expr=args.filter_expr,
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        conditioning_factor=args.conditioning_factor,
        random_slopes=not args.no_random_slopes,
        factor_indices=args.factor_indices,
    )
