from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
from torch import Tensor, cat


class OutputSelector(ABC):
    def __call__(self, model_outputs: Any) -> Any:
        return self.select(model_outputs)

    @abstractmethod
    def select(self, model_outputs: Any) -> Any:
        pass


class IndexSelector(OutputSelector):
    def __init__(self, idx: Union[int, List[int]] = 0) -> None:
        if isinstance(idx, int):
            idx = [idx]
        self.idx = idx

    def select(self, model_outputs):
        out = model_outputs
        for i in self.idx:
            out = out[i]
        return out


def create_output_transform(pred_transform):
    def output_transform(output):
        pred, target = output
        pred = pred_transform(pred)
        return pred, target
    return output_transform


SLOT_AE_OUTPUT = Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]

def collate_slotae_output(slot_ae_outputs: List[SLOT_AE_OUTPUT]) -> SLOT_AE_OUTPUT:
    recons, seg_mask, slots, attn_mask = [], [], [], []

    for (r, sm), (s, am) in slot_ae_outputs:
        recons.append(r)
        seg_mask.append(sm)
        slots.append(s)
        attn_mask.append(am)

    return (cat(recons), cat(seg_mask)), (cat(slots), cat(attn_mask))
