from typing import Any, Dict, List, Optional, Type
from functools import partial
from torch.utils.data import Dataset


class OODLoader:
    def __init__(self,
        dataset_cls: Type[Dataset],
        data_path: str,
        condition: Optional[str] = None,
        variant: Optional[str] = None,
        modifiers: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str,Any]] = None,
    ) -> None:
        assert condition is None or variant is not None

        if dataset_params is None:
            dataset_params = {}

        self.dataset_cls = dataset_cls
        self.data_path = data_path
        self.condition = condition
        self.variant = variant
        self.modifiers = modifiers
        self.dataset_params = dataset_params

        split_fns = self.get_splits()
        self.split_train, self.split_test = split_fns

    def load_dataset(self, stage):
        if stage == "fit":
            split_fn = self.split_train
        elif stage in ["test", "predict"]:
            split_fn = self.split_test
        elif stage == "all":
            split_fn = None
        else:
            raise ValueError(f"Not recognized stage {stage}")

        return self.dataset_cls(
            self.data_path,
            split_fn,
            **self.dataset_params,
        )

    def get_splits(self):
        dataset_cls = self.dataset_cls
        all_splits = dataset_cls.get_splits()
        all_mods = dataset_cls.get_modifiers()

        try:
            if self.condition is None:
                masks = None, None
            else:
                masks = all_splits[self.condition][self.variant]
        except KeyError:
            raise ValueError(
                f"Unrecognized variant {self.variant} for condition {self.condition}"
            )

        if self.modifiers is not None:
            for mod in self.modifiers:
                if mod not in all_mods:
                    raise ValueError('Unrecognized modifier {}'.format(mod))

                # If no mask, then modifier is only mask and
                # it is applied during training.
                if masks[0] is None:
                    masks = all_mods[mod], None
                else:
                    modf = partial(compose, mod=all_mods[mod])
                    masks = [(None if m is None else modf(m)) for m in masks]

        return masks

def compose(mask, mod):
    def composed(factor_values, factor_classes):
        return (mask(factor_values, factor_classes) &
                mod(factor_values, factor_classes))
    return composed
