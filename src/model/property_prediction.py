from typing import Callable, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as lsa

from src.model.base import BaseModel, TrainingInit
from src.model.slotae import SlotAutoEncoder
from src.model.autoencoder import VariationalAutoEncoder


class AutoencoderPropertyPrediction(BaseModel):
    def __init__(
        self,
        latent_size: int,
        n_properties: int,
        autoencoder: VariationalAutoEncoder,
        prediction_net: nn.Module,
        loss: Callable,
        training: TrainingInit
    ) -> None:
        super().__init__(training)
        autoencoder.freeze()

        self.n_properties = n_properties
        self.latent_size = latent_size
        self.autoencoder = autoencoder
        self.prediction_net = prediction_net
        self.loss = loss

    def forward(self, inputs):
        autoencoder_output = self.autoencoder(inputs)
        latents = autoencoder_output[1].detach()
        prediction = self.prediction_net(latents).sigmoid()
        return prediction.unsqueeze(1), autoencoder_output

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch

        assert targets.shape[-1] == self.n_properties

        prediction, _ = self.forward(inputs)
        cost = slot_assignment(prediction, targets, self.loss)[1]

        # loss = self.loss(prediction, targets)
        loss = cost.sum() / len(targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss


class SlotPropertyPrediction(BaseModel):
    """
    Perform property prediction on the slot representations generated by
    a Slot Attention module. Will perform set matching using the Linear Sum
    Assignment algorithm from Scipy.
    """
    def __init__(
        self,
        slot_size: int,
        n_properties: int,
        slot_ae: SlotAutoEncoder,
        prediction_net: nn.Module,
        loss: Callable,
        training: TrainingInit,
    ):
        super().__init__(training)
        slot_ae.freeze()

        self.n_properties = n_properties
        self.slot_size = slot_size
        self.slot_ae = slot_ae
        self.prediction_net = prediction_net
        self.loss = loss

    def forward(self, inputs):
        slot_ae_output = self.slot_ae(inputs)
        slots = slot_ae_output[1][0].detach()
        prediction = self.prediction_net(slots).sigmoid()
        return prediction, slot_ae_output

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"]
    ):
        inputs, targets = batch

        assert targets.shape[-1] == self.n_properties

        prediction, _ = self.forward(inputs)
        cost = slot_assignment(prediction, targets, self.loss)[1]

        # loss = self.loss(prediction, targets)
        loss = cost.sum() / len(targets)

        is_train = phase == "train"
        self.log(
            f"{phase}/loss",
            loss,
            on_epoch=not is_train,
            on_step=is_train,
            prog_bar=is_train,
            sync_dist=not is_train,
            rank_zero_only=True
        )

        return loss


class PropertyPrediction:
    """
    Used to measure the specific performance when predicting a particular
    property when analysing the model. It works as a wrapper that processes
    the outputs of the SlotPropertyPrediction.
    """
    def __init__(self, assignment_cost, target_object) -> None:
        self.assignment_cost = assignment_cost
        self.target_object = target_object

    def __call__(self, outputs) -> Tuple[torch.Tensor, torch.Tensor]:
        (slot_pred, _), targets = outputs

        targets = targets[:, self.target_object: self.target_object + 1]

        assigned_pred = slot_assignment(
            slot_pred, targets, self.assignment_cost)[0].squeeze(1)

        targets = targets.squeeze(1).argmax(-1)

        return assigned_pred, targets


ASSIGNMENT_COST_FN = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def slot_assignment(
    slot_pred: torch.Tensor,
    targets: torch.Tensor,
    assignment_cost: ASSIGNMENT_COST_FN,
):
    B = len(slot_pred)

    targets = targets.unsqueeze(1).expand(-1, slot_pred.size(1), -1, -1)
    slot_pred = slot_pred.unsqueeze(2).expand(-1, -1, targets.size(2), -1)

    pairwise_cost = assignment_cost(slot_pred , targets).sum(-1)

    # with mp.Pool(10) as p:
    #     assignment = p.map(lsa, pairwise_cost.detach().tolist())
    assignment = map(lsa, pairwise_cost.detach().tolist())

    input_idx, target_idx = tuple(zip(*assignment))
    input_idx = np.array(input_idx)
    target_idx = np.array(target_idx)

    batch_idx = torch.arange(B).unsqueeze_(-1)

    return (
        slot_pred[batch_idx, input_idx, target_idx],
        pairwise_cost[batch_idx, input_idx, target_idx],
        (batch_idx, input_idx, target_idx)
    )
