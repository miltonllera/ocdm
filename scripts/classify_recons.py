import numpy as np
import torch
import logging
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from src.model.vae import BetaVAE, WassersteinMMDAE
from src.model.slot import SlotAutoencoder, FigureGroundAutoencoder
from src.model.regression import Regressor, AdversarialDiscriminator
from src.dataset.dsprites import DSprites
from src.dataset.shapes3d import Shapes3D
from src.dataset.pentominos import Pentominos
from src.dataset.datamodule import DisentangledDataModule


torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


datasets = {
    'dsprites': (DSprites, "data/datasets/dsprites/dsprites_train.hdf5"),
    'shapes3d': (Shapes3D, "data/datasets/shapes3d/3dshapes.h5"),
    'pentominos': (Pentominos, "data/datasets/pentominos"),
}


model_types = {
    'vae': BetaVAE,
    'wae': WassersteinMMDAE,
    'sa1': FigureGroundAutoencoder,
    'sa': SlotAutoencoder,
}


def main(
    dataset_name: str,
    filter_expr: str,
    model_name: str,
    model_checkpoint: Path,
    classifier_type: str,
    classifier_path: Path,
):
    # Load data
    _logger.info("Loading data...")
    datamodule = DisentangledDataModule(
        datasets[dataset_name][0],
        "classification",
        held_out_filter=filter_expr,
        batch_size=1,
        val_split=0.0,
        test_split=0.0,
        path=datasets[dataset_name][1]
    )
    train_loader = datamodule.setup('fit').train_dataloader()
    test_loader = datamodule.setup('test').test_dataloader()

    # Load model checkpoint
    _logger.info("Loading model...")
    model = model_types[model_name].load_from_checkpoint(model_checkpoint)

    # Get classifier
    _logger.info("Loading classifier...")
    if classifier_type == 'factor_prediction':
        classifier = Regressor.load_from_checkpoint(classifier_path)
    elif classifier_type == 'discriminator':
        classifier = AdversarialDiscriminator.load_from_checkpoint(classifier_path)
    else:
        raise RuntimeError()

    _logger.info("Setting up evaluation...")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    def eval_model(loader):
        xent, acc = [], []
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            crit, metr = classifier.eval_model(x, y, model)
            xent.append(crit.cpu().numpy())
            acc.append(metr.cpu().numpy())
        return np.asarray(xent), np.asarray(acc)

    _logger.info("Evaluating model")
    train_xent, train_acc = eval_model(train_loader)
    test_xent, test_acc = eval_model(test_loader)

    _logger.info("Saving results...")
    np.savez(
        model_checkpoint.parent / f"{classifier_type}_scores.npz",
        train_xent=train_xent,
        train_acc=train_acc,
        test_xent=test_xent,
        test_acc=test_acc,
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--filter_expr", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_checkpoint", type=Path)
    parser.add_argument("--classifier_type", type=str)
    parser.add_argument("--classifier_path", type=Path)

    main(**vars(parser.parse_args()))
