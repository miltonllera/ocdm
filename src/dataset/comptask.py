import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class CompositionTask(Dataset):
    def __init__(self, dataset, seed: int | None = None):
        self.dataset = dataset
        self.index_map = IndexMap(dataset)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        z_code = self.dataset.factor_classes[idx]  # type: ignore
        transf_z_code = z_code.copy()

        # select factor to transform and value to transform to
        all_factors = np.arange(self.dataset.n_factors)  # type: ignore
        self.rng.shuffle(all_factors)

        # Iterate through all dimensions until we sample a new value
        dim = None
        for dim in all_factors:
            new_dim_code = self.sample_factor(z_code, dim)

            # Only assign if we could sample a new value for that dimension
            # if it's the last dimension, then we don't have any other choice
            if (new_dim_code != transf_z_code[dim]) or (dim == all_factors[-1]):
                transf_z_code[dim] = new_dim_code
                break

        # sample a command image
        command_z_code = transf_z_code.copy()
        for d in range(len(self.dataset.factor_sizes)):  # type: ignore
            if d != dim:
                command_z_code[d] = self.sample_factor(command_z_code, d)

        action = one_hot(
            torch.LongTensor([dim]),
            num_classes=self.dataset.n_factors  # type: ignore
        ).squeeze()

        img = self.dataset[idx][0]
        command_img = self.code_to_image(command_z_code)
        transformed_img = self.code_to_image(transf_z_code)

        input_imgs = torch.stack([img, command_img], dim=0).contiguous()
        target = torch.stack(
            [img, command_img, transformed_img],
            dim=0,
        ).contiguous()

        return (input_imgs, action), target

    def sample_factor(self, factor_code, dim):
        factor_d_code = np.arange(self.dataset.factor_sizes[dim])

        # Determine which codes are valid
        possible_codes = np.repeat(
            factor_code[None],
            torch.asarray([len(factor_d_code)]),
            axis=0,
        )
        possible_codes[:, dim] = factor_d_code

        idxs = self.code_to_index(possible_codes)
        is_valid = self.index_map[idxs] != -1

        # If more than one value is valid, remove the current one
        if sum(is_valid) > 1:
            is_valid[factor_code[dim]] = False
        # else return current
        else:
            return factor_code[dim]

        prob = np.ones(self.dataset.factor_sizes[dim]) / (sum(is_valid))
        prob[~is_valid] = 0

        return self.rng.choice(factor_d_code, p=prob)

    def code_to_index(self, code):
        return self.index_map.index(code)

    def code_to_image(self, code):
        idx = self.index_map[self.code_to_index(code)]
        return self.dataset[idx][0]


class IndexMap:
    """
    Index map that for a given index in the full dataset, returns the corresponding
    index after applying a filter that excludes some factor combinations.

    Use this to sample relevant combinations the composition task.
    """
    def __init__(self, dataset):
        total_combs = np.prod(dataset.factor_sizes)

        self.code_bases = total_combs / np.cumprod(dataset.factor_sizes)

        if total_combs == len(dataset):
            index_table = None
        else:
            index_table = np.zeros(np.prod(dataset.factor_sizes),
                                   dtype=np.int64) - 1

            for i, c in enumerate(dataset.factor_classes):
                index_table[self.index(c)] = i

        self.index_table = index_table

    def __getitem__(self, item):
        if self.index_table is None:
            return item
        return self.index_table[item]

    def index(self, code):
        return np.asarray(np.dot(code, self.code_bases), np.int64)

    def is_valid(self, code):
        return self[self.index(code)] != -1
