import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from typing import Literal

from src.model.base import BaseModel, TrainingInit
from src.nn.utils.parsing import create_sequential


class Regressor(BaseModel):
    def __init__(self,
        input_size: tuple[int, ...],
        backbone_config: list,
        n_targets: int,
        training: TrainingInit,
        regression_type: Literal['binary', 'multiclass', 'continuous'],
        prediction_dim: int | None,
    ) -> None:
        super().__init__(training)
        self.save_hyperparameters()

        if regression_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
            metric = Accuracy(task='binary')
        elif regression_type == 'multiclass':
            criterion = nn.CrossEntropyLoss()
            metric = Accuracy(task='multiclass', num_classes=n_targets)
        elif regression_type == 'continuous':
            metric = criterion = nn.MSELoss()
        else:
            raise RuntimeError()

        backbone = create_sequential(input_size, backbone_config)
        head_input_shape = backbone(torch.zeros(input_size)[None]).shape[1]
        n_outputs = 1 if regression_type == 'binary' else n_targets
        output_head = nn.Linear(head_input_shape, n_outputs)

        self.backbone = backbone
        self.criterion = criterion
        self.metric = metric
        self.regression_type = regression_type
        self.prediction_dim = prediction_dim
        self.output_head = output_head
        self.n_classes = n_outputs

    def forward(self, inputs):
        return self.output_head(self.backbone(inputs))

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        x, y = batch

        if self.prediction_dim is not None:
            y = y[:, self.prediction_dim]

        prediction = self.forward(x)
        loss = self.criterion(prediction, y)

        is_train = phase == "train"
        if is_train:
            to_log = {'train/loss': loss}
        else:
            to_log = {
                'val/loss': loss,
                'val/accuracy': self.metric(prediction, y)
            }

        self.log_dict(
            to_log,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def eval_model(self, x, y, model):
        recons = model.reconstruction(x)
        y = y[:, self.prediction_dim]
        prediction = self(recons)
        xent = self.criterion(prediction, y)
        acc = self.metric(prediction, y)
        return xent, acc


class AdversarialDiscriminator(BaseModel):
    def __init__(self,
        input_size: tuple[int, int, int],
        model_class: BaseModel,
        model_ckpt: str,
        backbone_config: list,
        training: TrainingInit,
        rng: np.random.Generator | int = np.random.default_rng()
    ) -> None:
        super().__init__(training)
        self.save_hyperparameters()

        backbone = create_sequential(input_size, backbone_config)
        head_input_shape = backbone(torch.zeros(input_size)[None]).shape[-1]
        output_head = nn.Linear(head_input_shape, 1)

        self.backbone = backbone
        self.base_model = model_class.load_from_checkpoint(model_ckpt).requires_grad_(False)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = Accuracy('binary')
        self.output_head = output_head
        self.rng = np.random.default_rng(rng) if isinstance(rng, int) else rng

    def forward(self, inputs):
        return self.output_head(self.backbone(inputs))

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        x, _ = batch
        is_real = torch.asarray(
            self.rng.choice([False, True], size=(len(x),), p=(0.5, 0.5)),
            device=x.device,
        )
        x = torch.where(
            is_real[..., None, None, None],  # broadcast to image shape
            x,
            self.base_model.reconstruction(x)  # type: ignore
        )
        y = torch.where(
            is_real,
            torch.ones(len(x), device=x.device, dtype=torch.float32),
            torch.zeros(len(x), device=x.device, dtype=torch.float32),
        )[..., None]

        prediction = self.forward(x)
        loss = self.criterion(prediction, y)

        is_train = phase == "train"
        if is_train:
            to_log = {'train/loss': loss}
        else:
            to_log = {
                'val/loss': loss,
                'val/accuracy': self.metric(prediction, y)
            }

        self.log_dict(
            to_log,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss

    def eval_model(self, x, y, model):
        recons = model.reconstruction(x)
        y = torch.zeros(len(y), device=y.device, dtype=torch.float32)[..., None]
        prediction = self(recons)
        xent = self.criterion(prediction, y)
        acc = self.metric(prediction, y)
        return xent, acc
