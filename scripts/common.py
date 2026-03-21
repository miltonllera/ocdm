import logging
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.vae import BetaVAE, WassersteinMMDAE
from src.model.slot import SlotAutoencoder, FigureGroundAutoencoder
from src.dataset.dsprites import DSprites
from src.dataset.shapes3d import Shapes3D
from src.dataset.pentominos import Pentominos
from src.dataset.utils import build_filter


torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


DATASETS = {
    'dsprites': (DSprites, "data/datasets/dsprites/dsprites_train.hdf5"),
    'shapes3d': (Shapes3D, "data/datasets/3dshapes/3dshapes.h5"),
    'pentominos': (Pentominos, "data/datasets/pentominos"),
}

MODEL_TYPES = {
    'vae': BetaVAE,
    'wae': WassersteinMMDAE,
    'sa1': FigureGroundAutoencoder,
    'sa': SlotAutoencoder,
}


def compute_embeddings(model, loader, device):
    latents = []
    for x, _ in tqdm(loader, desc="Computing embeddings"):
        latents.append(model.embed(x.to(device)).cpu().numpy())
    return np.concatenate(latents, axis=0).squeeze()


def load_or_compute_embeddings(
    embeddings_path: Path,
    model,
    loader,
    device,
    force_recompute: bool = False,
    split_name: str = "train"
):
    key = f"{split_name}_embeddings"

    if embeddings_path.exists() and not force_recompute:
        _logger.info(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path)
        if key in data:
            return data[key].squeeze()
        if 'train_embeddings' in data and split_name == 'train':
            return data['train_embeddings'].squeeze()

    _logger.info(f"Computing {split_name} embeddings...")
    embeddings = compute_embeddings(model, loader, device)

    existing_data = {}
    if embeddings_path.exists():
        existing_data = dict(np.load(embeddings_path))

    existing_data[key] = embeddings
    _logger.info(f"Saving embeddings to {embeddings_path}")
    np.savez(embeddings_path, **existing_data)

    return embeddings


def load_model(model_name: str, checkpoint_path: Path, compile=True):
    model = MODEL_TYPES[model_name].load_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return torch.compile(model), device


def load_dataset(dataset_name: str, filter_expr: str, batch_size: int = 64, num_workers: int = 4):
    dataset_class, path = DATASETS[dataset_name]
    held_out_filter = build_filter(dataset_class, filter_expr)
    train_filter = lambda x: ~held_out_filter(x)

    train_dataset = dataset_class(path, data_filter=train_filter)
    test_dataset = dataset_class(path, data_filter=held_out_filter)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def load_vlm(model_id, device, compile=True):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id, device_map=device, attn_implementation="sdpa")
    if compile:
        model = torch.compile(model)
    return model, processor


def tensor_to_pil(tensor):
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    img = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def get_shape_name(dataset, shape_value):
    return dataset.map_shapes(np.array([shape_value]))[0]
