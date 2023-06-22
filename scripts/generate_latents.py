import numpy as np
import logging
from argparse import ArgumentParser
from pathlib import Path

from scripts.common import (
    DATASETS,
    MODEL_TYPES,
    compute_embeddings,
    load_model,
    load_dataset
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
):
    _logger.info("Loading data...")
    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        dataset_name, filter_expr, batch_size=1
    )

    _logger.info("Loading model...")
    model, device = load_model(model_name, model_checkpoint)

    _logger.info("Computing embeddings...")
    train_embeddings = compute_embeddings(model, train_loader, device)
    test_embeddings = compute_embeddings(model, test_loader, device)

    _logger.info("Saving results...")
    np.savez(
        model_checkpoint.parent / "embeddings.npz",
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings
    )

    _logger.info("Done!")


if __name__ == '__main__':
    parser = ArgumentParser()

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

    main(**vars(parser.parse_args()))
