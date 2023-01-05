from typing import Any, Dict, List, Optional
from functools import partial
from . import dsprites, shapes3d, mpi


class OODLoader:
    def __init__(self,
        dataset_name: str,
        data_path: str,
        condition: Optional[str] = None,
        variant: Optional[str] = None,
        modifiers: Optional[List[str]] = None,
        dataset_params: Optional[Dict[str,Any]] = None,
    ) -> None:
        assert condition is None or variant is not None

        if dataset_params is None:
            dataset_params = {}

        self.dataset_name = dataset_name
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
        else:
            split_fn = self.split_test

        dataset_cls = self.get_dataset()
        raw = dataset_cls.load_raw(self.data_path, split_fn)
        return dataset_cls(
            *raw,
            **self.dataset_params,
        )

    def get_dataset(self):
        name = self.dataset_name
        if name == "dsprites":
            return dsprites.DSprites
        elif name == "3dshapes":
            return shapes3d.Shapes3D
        elif name == "mpi3d":
            return mpi.MPI3D
        else:
            raise ValueError(f"Unrecognized dataset {name}")

    def get_splits(self):
        if self.dataset_name == "dsprites":
            all_splits, all_mods = dsprites.splits, dsprites.modifiers
        elif self.dataset_name == "3dshapes":
            all_splits, all_mods = shapes3d.splits, shapes3d.modifiers
        elif self.dataset_name == "mpi3d":
            all_splits, all_mods = mpi.splits, mpi.modifiers
        else:
            raise ValueError(f"Unrecognized dataset {self.dataset_name}")

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
