import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from src.dataset.pentominos import Pentominos


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def sample_shape_examples(dataset, n_per_shape, seed=42):
    rng = np.random.default_rng(seed)
    shape_idx = list(dataset.factors).index('shape')
    shape_values = dataset.factor_values[:, shape_idx]
    unique_shapes = np.unique(shape_values)
    examples = {}
    for shape_val in unique_shapes:
        name = dataset.map_shapes(np.array([shape_val]))[0]
        indices = np.where(shape_values == shape_val)[0]
        chosen = rng.choice(indices, size=n_per_shape, replace=False)
        examples[name] = [tensor_to_pil(dataset[int(i)][0]) for i in chosen]
    return examples


def compose_example_grid(examples, test_image):
    w, h = test_image.size
    n = len(examples)
    grid = Image.new('RGB', (n * w + 2 + w, h), color=(128, 128, 128))
    for i, img in enumerate(examples):
        grid.paste(img.convert('RGB'), (i * w, 0))
    grid.paste(test_image.convert('RGB'), (n * w + 2, 0))
    return grid


def build_prompt_template(dataset, query_factor, n_examples=0):
    if dataset == 'dsprites':
        template = "A {rotated} white {shape_name} on a black background."
        def prompt_builder(shape_name, orientation, **kwargs):
            return template.format(
                shape_name=shape_name,
                rotated="rotated" if orientation > 0.45 or orientation > 5.9 else ""  # radians
            )
    elif dataset == "shapes3d":
        template = "A scene with a {color} colored {shape_name} at the center."
        def prompt_builder(shape_name, color, **kwargs):
            return template.format( shape_name=shape_name, color=color)
    elif dataset == 'pentominos':
        if n_examples > 0:
            def prompt_builder(shape_name, angle, **kwargs):
                return (
                    f"The {n_examples} leftmost white shapes on a black background are rotated "
                    f"examples of the {shape_name} pentomino. Is the rightmost white shape on a "
                    f"black background also an example of a rotated {shape_name} pentomino?"
                )
        else:
            template = "A {rotated} white {shape_name} on a black background."
            def prompt_builder(shape_name, angle, **kwargs):
                return template.format(
                    shape_name=shape_name,
                    rotated="rotated" if angle > 20 or angle < 340 else ""
                )
    else:
        raise RuntimeError("Unrecognized dataset")

    return prompt_builder


def query_vlm_batch(images, prompts, vlm, processor, device):
    inputs = processor(
        text=prompts,
        images=images,
        padding='max_length',
        max_length=64,
        return_tensors="pt"
    ).to(device)
    outputs = vlm(**inputs)
    probs = torch.sigmoid(outputs.logits_per_image)
    return probs > 0.5, probs


def evaluate_split(
    dataset,
    loader,
    model,
    prompt_builder,
    vlm,
    processor,
    device,
    query_factor,
    shape_examples=None,
):
    factor_idx = list(dataset.factors).index(query_factor) if query_factor is not None else None
    shape_idx = list(dataset.factors).index('shape')

    yes_count = no_count = 0
    details = []
    sample_offset = 0

    # if loader is not None:
    for x, _ in tqdm(loader, desc="Evaluating"):
        recons = model.reconstruction(x.to(device)).cpu()
        batch_size = x.shape[0]

        pil_images, prompts, shape_names, factor_vals_list = [], [], [], []

        for i in range(batch_size):
            sample_idx = sample_offset + i
            factor_vals = dataset.factor_values[sample_idx]
            shape_val = factor_vals[shape_idx]
            factor_val = factor_vals[factor_idx] if factor_idx is not None else None
            name = get_shape_name(dataset, shape_val)
            if shape_examples is not None:
                pil_images.append(compose_example_grid(shape_examples[name], tensor_to_pil(recons[i])))
            else:
                pil_images.append(tensor_to_pil(recons[i]))
            prompts.append(prompt_builder(name, factor_val))
            shape_names.append(name)
            factor_vals_list.append(factor_val)

        responses, scores = query_vlm_batch(pil_images, prompts, vlm, processor, device)

        for i, (res, p, name, factor_val, query) in enumerate(
            zip(responses, scores, shape_names, factor_vals_list, prompts)
        ):
            details.append({
                'sample_idx': sample_offset + i,
                'shape': name,
                'factor_value': factor_val,
                'prompt': query,
                'answer': 'yes' if res[i].item() else 'no',
                'score': p[i].item(),
            })
            yes_count += res[i].item()
            no_count += (~res[i]).item()

        sample_offset += batch_size

    total = yes_count + no_count
    accuracy = yes_count / total if total > 0 else 0.0
    return accuracy, yes_count, no_count, total, details


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    query_factor: str,
    vlm_model: str,
    batch_size: int = 128,
    n_examples: int = 0,
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
    prompt_builder = build_prompt_template(dataset_name, query_factor, n_examples)

    shape_examples = None
    if dataset_name == 'pentominos' and n_examples > 0:
        _, path = DATASETS['pentominos']
        ref_dataset = Pentominos(path)
        shape_examples = sample_shape_examples(ref_dataset, n_examples)

    _logger.info("Evaluating train split...")
    train_acc, train_yes, train_no, train_total, train_details = evaluate_split(
        train_dataset, train_loader, model, prompt_builder, vlm, processor, device, query_factor,
        shape_examples=shape_examples,
    )

    _logger.info("Evaluating test split...")
    test_acc, test_yes, test_no, test_total, test_details = evaluate_split(
        test_dataset, test_loader, model, prompt_builder, vlm, processor, device, query_factor,
        shape_examples=shape_examples,
    )

    print("=== VLM Evaluation Results ===")
    print(f"Train — accuracy: {train_acc:.4f}, yes: {train_yes}, no: {train_no}, "
          f"total: {train_total}")
    print(f"Test  — accuracy: {test_acc:.4f}, yes: {test_yes}, no: {test_no}, "
          f"total: {test_total}")

    output_dir = model_checkpoint.parent

    summary_path = output_dir / "vlm_eval.csv"
    _logger.info(f"Saving summary to {summary_path}")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "accuracy", "yes", "no", "total"])
        writer.writerow(["train", f"{train_acc:.4f}", train_yes, train_no, train_total])
        writer.writerow(["test", f"{test_acc:.4f}", test_yes, test_no, test_total])

    detail_path = output_dir / "vlm_eval_detail.csv"
    _logger.info(f"Saving details to {detail_path}")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_idx", "shape", "factor_value", "prompt", "answer", "score"])
        for row in train_details:
            writer.writerow(["train", row['sample_idx'], row['shape'], row['factor_value'],
                             row['prompt'], row['answer'], row['score']])
        for row in test_details:
            writer.writerow(["test", row['sample_idx'], row['shape'], row['factor_value'],
                             row['prompt'], row['answer'], row['score']])

    _logger.info("Done!")


if __name__ == '__main__':
    parser = ArgumentParser(description="VLM-based evaluation of image reconstructions")

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
        help="Factor to query (e.g., orientation, object_hue)")
    parser.add_argument("--vlm_model", type=str, default="google/siglip2-so400m-patch14-384",
        help="HuggingFace model ID for VLM")
    parser.add_argument("--batch_size", type=int, default=128,
        help="Batch size for reconstruction")
    parser.add_argument("--n_examples", type=int, default=0,
        help="Number of example ground-truth images per shape for visual prompting (pentominos only)")

    args = parser.parse_args()
    main(**vars(args))
