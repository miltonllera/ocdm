import numpy as np
import torch
from torch.utils.data import Dataset


class ColoredMultiShapeDataset(Dataset):
    """
    A dataset wrapper that samples multiple shapes from a base dataset (e.g. Pentominos or DSprites),
    adds discrete color, prevents shape overlap, and allows filtering out specific combinations.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        n_shapes: int = 2,
        num_colors: int = 10,
        filter_fn = None,
        prediction_type: str = 'unsupervised',
        prevent_overlap: bool = False,
        seed: int | None = None,
    ):
        self.base_dataset = base_dataset
        self.n_shapes = n_shapes
        self.num_colors = num_colors
        self.filter_fn = filter_fn
        self.prediction_type = prediction_type
        self.prevent_overlap = prevent_overlap
        self.rng = np.random.default_rng(seed)

        # Define 10 default distinct colors, fallback to generating more if needed
        self.colors = self._get_distinct_colors(num_colors)

        # Pentominos has a 'color' factor of size 1. DSprites does not have 'color'.
        # We align both by tracking the index of 'color' in the final factors list.
        self.base_factors = list(base_dataset.factors)  # type: ignore
        if 'color' in self.base_factors:
            self.color_in_base = True
            self.color_idx_in_base = self.base_factors.index('color')
            self.factors = self.base_factors
        else:
            self.color_in_base = False
            self.color_idx_in_base = len(self.base_factors)
            self.factors = self.base_factors + ['color']

        # Get final factor sizes
        base_factor_sizes = list(base_dataset.factor_sizes)  # type: ignore
        if self.color_in_base:
            base_factor_sizes[self.color_idx_in_base] = num_colors
            self.factor_sizes = base_factor_sizes
        else:
            self.factor_sizes = base_factor_sizes + [num_colors]

        # Pre-filter all single-shape combinations of (base_idx, color_idx)
        # to ensure that we only sample valid configurations.
        if filter_fn is None:
            base_idxs = np.arange(len(base_dataset))
            color_idxs = np.arange(num_colors)
            self.valid_combinations = np.stack(
                np.meshgrid(base_idxs, color_idxs, indexing='ij'),
                axis=-1
            ).reshape(-1, 2)
        else:
            self.valid_combinations = []
            for base_idx in range(len(base_dataset)):
                base_code = base_dataset.factor_classes[base_idx].astype(np.int64)  # type: ignore
                for color_idx in range(num_colors):
                    if self.color_in_base:
                        colored_code = base_code.copy()
                        colored_code[self.color_idx_in_base] = color_idx
                    else:
                        colored_code = np.append(base_code, color_idx)

                    factor_dict = dict(zip(self.factors, colored_code))

                    if filter_fn(factor_dict):
                        self.valid_combinations.append((base_idx, color_idx))
            self.valid_combinations = np.array(self.valid_combinations)
        if len(self.valid_combinations) == 0:
            raise ValueError("Exclusion filter removed all possible combinations")

    def _get_distinct_colors(self, num_colors):
        # 10 distinct RGB colors
        palette = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 0.5],  # Purple
            [0.0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 0.5],  # Gray
        ]
        if num_colors <= len(palette):
            return torch.tensor(palette[:num_colors])

        # Fallback to generate more colors dynamically
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            h = hue * 6.0
            c = 1.0
            x = c * (1.0 - abs(h % 2.0 - 1.0))
            if h < 1:
                r, g, b = c, x, 0.
            elif h < 2:
                r, g, b = x, c, 0.
            elif h < 3:
                r, g, b = 0., c, x
            elif h < 4:
                r, g, b = 0., x, c
            elif h < 5:
                r, g, b = x, 0., c
            else:
                r, g, b = c, 0., x
            colors.append([r, g, b])
        return torch.tensor(colors)

    def __len__(self):
        return len(self.valid_combinations)

    def __getitem__(self, idx):
        # Use a local deterministic generator seeded with idx for thread safety and reproducibility
        local_rng = np.random.default_rng(idx)

        sampled_shapes = []
        sampled_masks = []
        sampled_factors = []

        # Always use self.valid_combinations[idx] as the first shape
        first_base_idx, first_color_idx = self.valid_combinations[idx]

        first_sample = self.base_dataset[first_base_idx]
        first_img = first_sample[0] if isinstance(first_sample, tuple) else first_sample
        if first_img.shape[0] > 1:
            first_img = first_img[:1]
        first_mask = first_img > 0.0

        sampled_shapes.append((first_img, first_color_idx))
        sampled_masks.append(first_mask)

        first_base_code = self.base_dataset.factor_classes[first_base_idx].astype(np.int64)  # type: ignore
        if self.color_in_base:
            first_code = first_base_code.copy()
            first_code[self.color_idx_in_base] = first_color_idx
        else:
            first_code = np.append(first_base_code, first_color_idx)
        sampled_factors.append(first_code)

        # Sample the remaining shapes, ensuring no overlap
        attempts = 0
        while len(sampled_shapes) < self.n_shapes and attempts < 300:
            attempts += 1
            rand_idx = local_rng.choice(len(self.valid_combinations))
            base_idx, color_idx = self.valid_combinations[rand_idx]

            sample = self.base_dataset[base_idx]
            img = sample[0] if isinstance(sample, tuple) else sample
            if img.shape[0] > 1:
                img = img[:1]
            mask = img > 0.0

            overlap = False
            if self.prevent_overlap:
                for existing_mask in sampled_masks:
                    if (mask & existing_mask).any():
                        overlap = True
                        break

            if not overlap:
                sampled_shapes.append((img, color_idx))
                sampled_masks.append(mask)

                base_code = self.base_dataset.factor_classes[base_idx].astype(np.int64)  # type: ignore
                if self.color_in_base:
                    code = base_code.copy()
                    code[self.color_idx_in_base] = color_idx
                else:
                    code = np.append(base_code, color_idx)
                sampled_factors.append(code)

        # Color and composite shapes
        colored_imgs = []
        for img, color_idx in sampled_shapes:
            color = self.colors[color_idx].view(3, 1, 1)
            colored_img = img * color
            colored_imgs.append(colored_img)

        combined_image = torch.stack(colored_imgs, dim=0).max(dim=0)[0]

        if self.prediction_type == 'unsupervised':
            return combined_image, combined_image

        all_factors = np.concatenate(sampled_factors)
        return combined_image, all_factors
